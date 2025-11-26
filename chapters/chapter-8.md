# Data Parallelism: Batch Sharding and Distributed Dataloaders

This chapter focuses on scaling training across multiple devices by replicating your model and splitting the data. You’ll learn how to turn a single-GPU or single-CPU script into a multi-device setup that still feels like “normal PyTorch/JAX/TensorFlow,” while avoiding common pitfalls around synchronization, randomness, and input pipelines.

Learning objectives:
- Explain the core ideas behind data parallelism and when it is a good fit.
- Distinguish between single-process and multi-process data parallel training.
- Implement batch sharding and per-device dataloaders in common frameworks.
- Reason about performance trade-offs: communication vs computation vs input pipeline.
- Diagnose and fix common bugs (data duplication, skew, poor utilization).

Sections:
- Conceptual overview: what is data parallelism?
- Single-node, multi-GPU data parallelism
- Sharding batches and sampling data correctly
- Distributed dataloaders and input pipelines
- Communication, synchronization, and performance
- Debugging and validation in distributed training
- Exercises: scaling a baseline script with data parallelism

## Conceptual overview: what is data parallelism?

At its core, data parallelism is the idea of *replicating* your model across multiple devices and feeding each replica a different slice of the input batch. Every replica computes forward and backward passes on its own mini-batch, then all replicas cooperate to average (or otherwise combine) their gradients so the model parameters stay in sync. Conceptually, you are still training a single model; you’re just processing more examples per step by doing the work in parallel.

It helps to contrast this with the “default” single-device setup. In a typical training loop, you:
- Take a batch of data.
- Run it through the model.
- Compute the loss and gradients.
- Update the parameters.

In a data-parallel setup, the loop looks almost the same, but the batch is first split across devices, each device runs those steps independently, and then a synchronization step merges the results before the optimizer update. Frameworks hide most of the low-level details, but the mental model—replicas plus gradient sync—remains the same across PyTorch, TensorFlow, JAX, and others.

There are a few key properties to keep in mind:

- **Replica semantics:** Each device holds a full copy of the model weights. If you inspect any replica after synchronization, it should match the others. Bugs often arise when some state is *not* replicated or synchronized (for example, random number generators, batch-norm statistics, or optimizer state).

- **Effective batch size:** When you split a batch across `N` devices, the *per-device* batch size shrinks, but the *global* batch size is the sum across all devices. This has direct consequences for learning rate schedules, convergence behavior, and memory usage. Many “my distributed run diverges” issues trace back to unintentional global batch size changes.

- **Communication vs computation:** As you scale to more devices, the cost of synchronizing gradients becomes a larger fraction of each step. On a single machine with fast interconnects, this might be negligible; across multiple nodes on Ethernet, it can dominate. Data parallelism works best when each device has enough computation to amortize the cost of communicating its gradients.

Finally, data parallelism assumes that your training examples are *independent and identically distributed* enough that shuffling and partitioning them across devices does not change the learning problem. This is usually true for standard supervised learning tasks, but less obvious for sequence training, curriculum learning, or non-iid data sources. Later sections will tackle how to shard data correctly and how distributed dataloaders keep replicas fed without silently skewing your dataset.

## Single-node, multi-GPU data parallelism

Most people first meet data parallelism on a single machine with more than one GPU. This setup is friendly: all devices share memory through the same host, the interconnect is relatively fast, and you can usually start from an existing single-GPU script.

The high-level pattern looks like this:

- You run one *process per GPU* (either launched manually or via a helper).
- Each process is responsible for one device and a shard of the global batch.
- Gradients are synchronized across processes at every optimization step.

Frameworks provide different entry points but share the same ideas. In PyTorch, this is `DistributedDataParallel` (DDP); in TensorFlow, `MirroredStrategy`; in JAX, `pmap`. In each case, you write a training step that looks almost like your single-GPU version, and the framework wraps it with the logic needed to talk to other devices.

A few practical aspects matter on a single node:

- **Process model vs threads:** Multi-GPU training is usually done with multiple *processes*, not threads, because frameworks rely on separate CUDA contexts per process. This means each process re-creates the model, dataloader, and optimizer—but with a specific device and rank.

- **Local vs global batch sizes:** If your original script used `batch_size=128` on one GPU and you now run on 4 GPUs, you should decide whether:
  - Each GPU still sees 128 examples (global batch becomes 512), or
  - You divide the batch so each GPU sees 32 examples (global batch stays 128).

  Both are valid, but they lead to different optimization behavior. The safest starting point is to keep the *global* batch size fixed and adjust the per-GPU batch accordingly.

- **Device affinity:** Each process should talk to exactly one GPU. On a single node, this is often as simple as:
  - Setting `CUDA_VISIBLE_DEVICES` per process, or
  - Using a rank-to-device mapping (`device_id = local_rank`).

To make this concrete, here is a stripped-down PyTorch sketch that shows structure rather than full error handling:

```python
# Single-node, multi-GPU skeleton with PyTorch DDP
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main_worker(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model = MyModel().to(device)
    model = DDP(model, device_ids=[device])

    # You’ll connect a DistributedSampler here in the next section
    train_loader = make_dataloader(...)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for batch in train_loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = compute_loss(out)
        loss.backward()
        optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # Typically launched via: torchrun --nproc_per_node=$WORLD_SIZE train.py
    rank = int(os.environ["RANK"])
    main_worker(rank, world_size)
```

You don’t need to memorize the exact API calls yet. The main takeaway is structural:

- The *training step* body is almost unchanged from single-GPU.
- The *wrapper* (process initialization, device assignment, DDP construction) is where distributed specifics live.
- On a single node, communication uses high-bandwidth links (PCIe/NVLink) and is usually fast enough that you can focus first on getting the logic correct.

Later sections will fill in the missing pieces—especially how to create dataloaders that cooperate with multiple processes so that each GPU sees the right shard of data without duplication. On a single machine, once those are in place, data parallelism often feels like the “easy mode” of distributed training.

## Sharding batches and sampling data correctly

Once you have one process per GPU, the next question is simple to ask and surprisingly subtle to answer: *which examples does each process see, and in what order?* Getting this wrong leads to silent bugs—duplicated data, missing data, skewed distributions—that can hurt convergence without throwing obvious errors.

There are three layers to think about:

1. **Dataset partitioning across processes.**
2. **Batch construction within each process.**
3. **Shuffling and epoch boundaries.**

On a single node with `world_size = N` processes, a common goal is:

- Across *all* processes, each example appears exactly once per epoch (or at least with well-understood frequency).
- Within each process, local batches are contiguous slices of that process’s shard.
- Shuffling is reproducible and independent across epochs.

If you’re using a modern framework, you rarely implement this logic from scratch; instead, you configure the provided samplers or input pipelines.

### The mental model

Imagine your dataset as a simple array of indices:

```text
[0, 1, 2, 3, 4, 5, 6, 7, ..., N-1]
```

For `world_size = 4`, you want process 0 to see one quarter of these indices, process 1 another quarter, and so on. Depending on your sampler, this can be done by:

- **Striding:** process `r` takes indices `r, r + world_size, r + 2*world_size, ...`
- **Chunking:** split the full list into 4 contiguous blocks and assign one block per rank.

Both can work; frameworks typically choose one and handle corner cases (like datasets whose size is not divisible by `world_size`).

Within each process, these indices are then grouped into batches of `batch_size_per_gpu`. The **global batch size** becomes:

```text
global_batch = batch_size_per_gpu * world_size
```

You should keep this formula in mind whenever you change `world_size` or `batch_size_per_gpu`.

### PyTorch: DistributedSampler and per-rank dataloaders

In PyTorch, `DistributedSampler` is the main tool for sharding datasets in data-parallel training. It cooperates with `DistributedDataParallel` by making sure each rank sees a distinct subset of data for each epoch:

```python
from torch.utils.data import DataLoader, DistributedSampler

def make_dataloader(dataset, batch_size, world_size, rank):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,      # or False for evaluation
        drop_last=False,   # often True for training if you want equal-sized batches
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    return loader, sampler
```

And in your main worker:

```python
def main_worker(rank, world_size):
    # ... init process group, create device and model ...
    dataset = MyDataset(...)
    train_loader, train_sampler = make_dataloader(
        dataset,
        batch_size=per_gpu_batch,   # per-process batch size
        world_size=world_size,
        rank=rank,
    )

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # important for proper shuffling
        for batch in train_loader:
            # normal training step on this rank
            ...
```

Key details:

- **`num_replicas` and `rank`:** These tell the sampler how many processes there are in total and which “slice” this process should handle.
- **`set_epoch(epoch)`:** Calling this once per epoch lets the sampler change its random seed so that shuffling differs between epochs but remains consistent across ranks. Forgetting this call is a common source of subtle data-order issues.
- **`drop_last`:** If `False`, the last batch may be smaller or imbalanced across ranks. If you rely on synchronized batch statistics (e.g., with some batch-norm configurations), using `drop_last=True` can simplify life.

### Handling non-divisible dataset sizes

Real datasets almost never have sizes that are a clean multiple of `(world_size * batch_size_per_gpu)`. Frameworks typically offer two approaches:

- **Pad:** artificially extend the dataset with extra samples (often by repeating the last few). This preserves equal-sized shards across ranks at the cost of seeing some examples slightly more often.
- **Uneven last batch:** allow some ranks to have one more element than others, or to stop one step earlier.

`DistributedSampler`’s `drop_last` and `shuffle` flags control this behavior. For most training scenarios, small padding is acceptable and simpler than trying to micro-optimize away a handful of repeats.

### Evaluation and validation splits

For validation or test sets, your goals change slightly:

- You often care more about *covering all examples* than about perfect shuffling.
- It is usually fine (and sometimes preferable) to avoid drop-last and keep deterministic ordering.

In those cases, you can:

- Reuse `DistributedSampler` but set `shuffle=False` and `drop_last=False`.
- Or, in smaller setups, let every rank evaluate on the full validation set and aggregate metrics only on rank 0 (trading extra compute for simplicity).

### Common failure modes

Watch out for these easy-to-miss issues:

- **No sampler at all:** If each process builds a plain `DataLoader` over the full dataset with `shuffle=True`, every GPU will see the *entire* dataset every epoch—your effective global batch size inflates by `world_size`, and you lose the benefit of more data per epoch.
- **Missing `set_epoch`:** Shuffling may be “frozen,” giving the same order every epoch or inconsistent orders across ranks, which can hurt convergence or reproducibility.
- **Mismatched `world_size` or `rank`:** If `num_replicas` doesn’t equal the actual number of processes, or if ranks are mis-assigned, some data shards may never be seen.

The main takeaway is that *data sharding is part of your model definition*. When you change the number of GPUs or nodes, you are implicitly changing the way your data is sampled. By treating samplers and loaders as first-class, testable components, you avoid many of the headaches that make distributed training feel “random” or unstable.

## Distributed dataloaders and input pipelines

Once you’re sharding data correctly, the next challenge is keeping accelerators *fed*. Many “my GPUs are only 30% utilized” complaints turn out to be input pipelines that can’t deliver batches fast enough. In a data-parallel setting, this problem is multiplied: every process needs its own stream of batches, and they all need to arrive on time.

A good distributed input pipeline has three goals:

1. **Parallelism:** multiple CPU workers prepare data while GPUs are training.
2. **Shard-awareness:** each worker only touches the subset of data for its rank.
3. **Overlap:** data loading, preprocessing, and transfer to device overlap with computation so that devices rarely wait idle.

### CPU workers and prefetching

Most frameworks let you specify some combination of:

- Number of CPU workers / threads.
- Buffer sizes (how many batches to keep in memory).
- Whether to pin memory for faster host→device transfers.

In PyTorch, this is all configured on the `DataLoader`:

```python
from torch.utils.data import DataLoader, DistributedSampler

def make_dataloader(dataset, batch_size, world_size, rank, shuffle=True):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,      # tune based on CPU cores and preprocessing cost
        pin_memory=True,    # improves H2D copy speed
        prefetch_factor=2,  # each worker preloads this many batches
        persistent_workers=True,  # avoid worker respawn each epoch
    )
    return loader, sampler
```

Key ideas:

- Increasing `num_workers` lets more batches be prepared in parallel, up to the point where you saturate CPU or disk.
- `prefetch_factor` and similar knobs hide I/O latency by preparing future batches while the current one is on the GPU.
- `persistent_workers=True` avoids expensive worker restarts at epoch boundaries, which matters more when you have many short epochs.

### Overlapping data transfer with computation

On the GPU side, you want host→device copies to run *asynchronously* so they overlap with compute:

```python
device = torch.device(f"cuda:{rank}")

for batch in train_loader:
    batch = batch.to(device, non_blocking=True)  # async copy from pinned memory
    out = model(batch)
    loss = compute_loss(out)
    loss.backward()
    optimizer.step()
```

The combination of:

- `pin_memory=True` in the `DataLoader`, and
- `non_blocking=True` in `.to(device, ...)`

lets the runtime schedule copies and kernels on different CUDA streams. You still need to synchronize at step boundaries (the framework or optimizer will do this), but you avoid unnecessary stalls between batches.

### Multi-node considerations

Everything so far works on a single node. When you add more nodes, two extra constraints appear:

- The **global** `world_size` now includes processes on all nodes.
- Each process must still create a *local* dataloader, but `num_replicas` in the sampler should be the global world size so that shards are unique across the entire cluster.

A minimal skeleton:

```python
# Pseudocode: multi-node initialization
def main_worker(rank, world_size):
    # rank is global rank: 0..world_size-1
    dist.init_process_group(
        backend="nccl",
        init_method="env://",  # or a rendezvous URL
        world_size=world_size,
        rank=rank,
    )

    dataset = MyDataset(...)
    per_gpu_batch = 32

    train_loader, train_sampler = make_dataloader(
        dataset,
        batch_size=per_gpu_batch,
        world_size=world_size,  # global!
        rank=rank,
        shuffle=True,
    )

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            # standard training step
            ...
```

Your cluster launcher (`torchrun`, SLURM, Kubernetes operator, etc.) is responsible for assigning unique global ranks and passing them via environment variables.

### TensorFlow and JAX input pipelines

Other frameworks follow the same principles but with different APIs:

- **TensorFlow** uses `tf.data.Dataset` pipelines and `tf.distribute.Strategy`. You typically build a `Dataset` once, then call `strategy.experimental_distribute_dataset(dataset)`, which creates a per-replica view, handles sharding, and prefetches to devices (`dataset.prefetch(tf.data.AUTOTUNE)`).
- **JAX** often uses `tf.data` or custom Python generators, combined with `jax.pmap`. You shard batches along the leading dimension so that `pmap` can place a slice on each device, and you prefetch to device-side buffers using utilities like `prefetch_to_device`.

The underlying ideas are identical: shard-aware iterators, background workers, and explicit prefetching to avoid starving accelerators.

### Diagnosing pipeline bottlenecks

A few quick checks tell you whether your dataloaders are the bottleneck:

- GPU utilization is low (e.g., 20–40%) even though the model is reasonably large.
- CPU usage spikes to 100% on a few cores during training.
- Simple logging around the training loop shows long gaps between batches.

Common fixes:

- Increase `num_workers` and prefetch/buffer sizes.
- Move heavy preprocessing (image decoding, augmentation) off the main training process and into the dataset or a separate preprocessing step.
- Cache preprocessed data (e.g., to LMDB, WebDataset/tar files, TFRecords) to reduce per-epoch work.

In practice, a solid distributed input pipeline is one of the highest-leverage optimizations you can make: it benefits every epoch, every model variant, and every scale-out experiment you run on top of it.

## Communication, synchronization, and performance

So far we’ve focused on *feeding* each replica with the right data. The other half of data parallelism is getting replicas to *agree* on parameter updates. That agreement happens through communication: gradients (or parameter deltas) are exchanged between processes every step. How efficiently you do this largely determines how well your training scales.

At a high level, each step of synchronous data parallelism looks like:

1. Each replica computes gradients on its local batch.
2. Gradients are combined across replicas (usually averaged).
3. Every replica applies the same optimizer update to its local copy of the weights.

The second step is where communication and synchronization live.

### All-reduce and friends

Most modern data parallel implementations are built on top of **collective communication** primitives: `all_reduce`, `broadcast`, `reduce_scatter`, and so on. You rarely call these directly, but they power constructs like PyTorch DDP and TensorFlow’s synchronous strategies.

The workhorse is `all_reduce`:

- Each rank starts with a tensor (its gradients).
- The system applies an operation (e.g., sum) across all ranks.
- Every rank ends up with the same result.

Pseudocode for a synchronous step:

```python
# On each rank:
grads = compute_gradients(batch)
all_reduce_in_place(grads, op="sum")
grads /= world_size
optimizer.step(grads)
```

In practice, communication libraries (NCCL, MPI, Gloo, vendor-specific stacks) implement `all_reduce` using ring, tree, or hybrid algorithms that trade off latency and bandwidth. On a single node with NVLink, this can be very fast; across nodes over Ethernet, it can easily dominate step time.

### Where synchronization happens in common frameworks

- **PyTorch DDP** hooks into `loss.backward()`: as each parameter’s gradient tensor is computed, DDP schedules an all-reduce for that tensor or a “bucket” of tensors. This overlaps some communication with remaining backprop computation.
- **TensorFlow `MirroredStrategy`** and other sync strategies typically run a fused all-reduce over gradients once per step.
- **JAX `pmap`** exposes collectives (`lax.pmean`, `lax.psum`) directly, which you insert into your training step to average gradients or parameters.

The important point: every step has at least one *global barrier* where all ranks must participate. If one rank is slow (data stall, OS interference, noisy neighbor), all others wait.

### Communication vs compute: simple scaling model

A crude but useful mental model for per-step time:

```text
step_time ≈ compute_time + communication_time
```

- `compute_time` is how long it takes each device to run forward + backward on its local batch.
- `communication_time` is how long it takes to exchange gradients (or parameters) across all devices.

When you increase the number of devices:

- `compute_time` per device usually goes **down** (smaller per-device batch) or stays similar (if you scale the global batch).
- `communication_time` tends to go **up** with:
  - Number of devices.
  - Size of the model (total gradient size).
  - Network latency and bandwidth characteristics.

You get good scaling when `communication_time` is a small fraction of `compute_time`. This is why:

- Big models on fast interconnects (NVLink, InfiniBand) tend to scale well.
- Tiny models or very small batches can scale *worse* as you add GPUs: the communication overhead dominates.

### Practical knobs for better performance

Most frameworks expose a few levers to tune communication:

- **Bucket sizes / gradient fusion:** Instead of launching an all-reduce for every small gradient tensor, frameworks group many tensors into larger “buckets.” Larger buckets reduce per-call overhead but can delay when communication starts.
  - In PyTorch DDP, this is `bucket_cap_mb`.
- **Overlap with computation:** Scheduling all-reduce operations as soon as gradients for a bucket are ready lets communication run concurrently with the rest of backprop. You get this automatically in DDP and similar systems, but extremely small models or custom autograd patterns can reduce overlap.
- **Mixed precision / gradient compression:** Using FP16/BF16 gradients or lossy compression (e.g., 8-bit gradients) shrinks the amount of data sent per step. This can cut communication time substantially, especially on slower networks.
- **Gradient accumulation:** Accumulating gradients over multiple local mini-batches before syncing reduces the frequency of all-reduces at the cost of a larger effective batch size and slightly staler gradients.

### Single-node vs multi-node behavior

On a **single node**, GPUs usually share:

- PCIe (or better, NVLink/NVSwitch) for peer-to-peer communication.
- Relatively low-latency links and high bandwidth.

Here, communication overhead is often modest for typical model sizes; you’ll see close to linear speedups up to the number of GPUs, provided your input pipeline keeps up.

Across **multiple nodes**, communication must cross the network fabric:

- If you have high-speed interconnects (InfiniBand, RoCE), you may still get good scaling.
- On commodity Ethernet, latency and bandwidth can severely limit throughput, especially for large models or many nodes.

Symptoms of network-bound training:

- GPU utilization dips during synchronization phases.
- Adding more nodes improves throughput less and less (or even makes it worse).
- Profilers show large chunks of time in collective communication kernels.

In such cases, strategies like **data-parallel within nodes + model/pipeline parallel across nodes**, or using **asynchronous** or **elastic** training schemes, can be more effective—but they also increase complexity.

### Simple sanity checks

To reason about communication and synchronization in your own setup:

- Measure **single-GPU** step time and compare to **multi-GPU** step time.
- Use built-in profilers (PyTorch/TensorFlow/JAX) and look specifically for:
  - All-reduce or collective kernels.
  - Time spent in communication backends (NCCL, MPI).
- Experiment with:
  - Varying batch size per GPU (changes compute/comm ratio).
  - Turning mixed precision on/off.
  - Changing number of GPUs or nodes.

You don’t need a perfect performance model; even rough measurements will tell you whether you are compute-bound (optimize kernels, model, or batch size) or communication-bound (optimize interconnect usage, bucket sizes, or scaling strategy). The next section will build on this by showing how to detect correctness bugs and subtle divergences that can arise once many replicas, samplers, and communication steps are all in play.

## Debugging and validation in distributed training

Data parallel training is supposed to be “just like single-GPU, only faster.” In reality, it adds enough moving parts—multiple processes, samplers, collectives—that it’s easy to drift away from your baseline without noticing. This section is about keeping you honest: making sure your multi-GPU/multi-node runs are actually doing the same experiment you think they are.

There are three big classes of issues to watch for:

1. **Numerical differences and divergence** (loss curves don’t match).
2. **Data path bugs** (duplication, missing examples, skew).
3. **State synchronization problems** (some replicas behave differently).

### Start from a known-good single-device baseline

Before debugging anything distributed, you need a clean reference:

- Fix a seed and make single-device training deterministic where practical.
- Save:
  - The exact code and hyperparameters.
  - A short training log (loss vs step/epoch, key metrics).
  - Optionally a small checkpoint.

When you enable data parallelism, your first job is to reproduce this behavior as closely as possible, *keeping the effective global batch size constant*.

A simple checklist for the first distributed run:

- Same model, optimizer, learning rate schedule, and loss.
- Same dataset and preprocessing.
- Global batch size matches the baseline.
- Only change: enabling DDP / MirroredStrategy / pmap and sharding the data.

If the new loss curve is wildly different from the single-device run, treat it as a bug until proven otherwise.

### Check that every sample is seen exactly once per epoch

Many distributed “mysteries” turn out to be bad sharding. A quick diagnostic pattern is to instrument your dataset or sampler to record which indices each rank sees.

For example, with PyTorch:

```python
# In your Dataset
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return {"idx": idx, "x": x, "y": y}

# In your training loop (debug mode only)
seen_indices = []

for batch in train_loader:
    seen_indices.extend(batch["idx"].tolist())
    # ... normal training step ...
```

You can run:

- Once with `world_size=1`.
- Once with `world_size>1`, collecting `seen_indices` on each rank.

On rank 0 (or after gathering across ranks), verify:

- The multiset of indices across *all* ranks per epoch matches `range(len(dataset))`.
- No excessive duplication or large gaps.

You don’t need to keep this code permanently; it’s a good temporary probe when you suspect sharding issues.

### Validate loss and gradients across ranks

If the data looks correct but training still behaves strangely, the next suspicion is synchronization.

Two simple checks:

1. **Per-rank loss logging**

   Log loss (and maybe a small validation metric) per rank for a few steps:

   ```python
   loss_val = float(loss.detach())
   print(f"[rank {rank}] step {step} loss={loss_val}")
   ```

   For synchronous data parallelism with identical initial seeds and data sharding, losses across ranks should be in the same ballpark. Large, systematic differences may indicate that some rank is seeing different data, using different hyperparameters, or missing gradient sync.

2. **Gradient consistency (spot check)**

   On a debug run, you can assert that a small subset of parameters has identical gradients across ranks immediately after backprop but before the optimizer step:

   ```python
   # Debug-only: check one parameter tensor
   with torch.no_grad():
       for name, p in model.named_parameters():
           if p.grad is None:
               continue
           # pick a small slice
           local_slice = p.grad.view(-1)[:8].clone()
           # all_gather this slice to compare across ranks
           tensor_list = [torch.zeros_like(local_slice) for _ in range(world_size)]
           dist.all_gather(tensor_list, local_slice)
           # simple consistency check on rank 0
           if rank == 0:
               for i, t in enumerate(tensor_list):
                   if not torch.allclose(tensor_list[0], t, atol=1e-5, rtol=1e-4):
                       print(f"Gradient mismatch on {name} between rank 0 and rank {i}")
           break
   ```

   You wouldn’t keep this in a hot path, but it’s a powerful way to catch misconfigured DDP wrappers or custom layers that bypass gradient synchronization.

### Randomness, seeds, and nondeterminism

Randomness shows up in:

- Weight initialization.
- Data shuffling and augmentations.
- Dropout, stochastic layers, some optimizers.

In distributed settings you typically want:

- **Global reproducibility:** reruns of the same configuration produce similar curves.
- **Per-rank variation where appropriate:** e.g., different random augmentations per rank.

Guidelines:

- Set seeds on every process:

  ```python
  import random, numpy as np, torch

  def set_seed(seed, rank=0):
      full_seed = seed + rank  # simple rank offset
      random.seed(full_seed)
      np.random.seed(full_seed)
      torch.manual_seed(full_seed)
      torch.cuda.manual_seed_all(full_seed)
  ```

- Let the distributed sampler’s `set_epoch(epoch)` handle epoch-wise reshuffling.
- Avoid mixing deterministic assumptions (e.g., strict sample ordering) with random augmentations unless you explicitly control both.

If you need strong determinism for debugging, many frameworks offer flags to enforce deterministic algorithms (with some performance trade-offs). Use them temporarily to narrow down issues.

### Watch for unsynchronized model or optimizer state

Common pitfalls:

- **Buffers or state not in `model.parameters()`**: e.g., manually-maintained running statistics, custom normalization layers, or latent state in modules.
- **Optimizer state divergence**: different learning-rate schedulers or incorrectly reloaded checkpoints per rank.

Defense-in-depth:

- Always wrap the *whole* model with the distributed wrapper (DDP, MirroredStrategy, pmap) rather than just parts.
- When loading from checkpoints, have rank 0 load the state and then broadcast it to others, or rely on the framework’s built-in distributed checkpointing.
- Log a small hash of model parameters on each rank after a known step and assert they match:

  ```python
  import hashlib

  def param_hash(model):
      h = hashlib.sha256()
      with torch.no_grad():
          for p in model.parameters():
              h.update(p.detach().cpu().numpy().tobytes())
      return h.hexdigest()[:16]

  if step % 100 == 0:
      h_local = param_hash(model)
      # gather hashes for debugging
      hashes = [None for _ in range(world_size)]
      dist.all_gather_object(hashes, h_local)
      if rank == 0:
          print("Param hashes:", hashes)
  ```

Identical hashes across ranks are a quick sanity check that synchronization is working.

### Keep a “small debug mode”

Finally, maintain a **small, fast configuration** of your experiment for debugging:

- Tiny dataset subset (a few hundred examples).
- Fewer epochs, smaller model, small batch sizes.
- Ability to run both single-GPU and multi-GPU versions in a minute or two.

When something looks off at scale, reproduce it in this small mode. It makes it much easier to:

- Add assertions and extra logging.
- Run under profilers.
- Iterate on fixes without burning cluster hours.

In the last section of this chapter, you’ll put these ideas together by taking a reference single-GPU script, scaling it out with data parallelism, and using these validation patterns to ensure that “faster” didn’t quietly turn into “different.”

## Exercises: scaling a baseline script with data parallelism

These exercises walk you through taking a small, single-device training script and scaling it up with data parallelism. The goal is to practice the mechanics—launching multiple processes, sharding data, checking correctness—without getting lost in a huge model.

### 1. Turn a single-GPU PyTorch script into DDP on one node

Start from a minimal single-GPU training loop (for example, a small CNN on CIFAR-10 or an MLP on MNIST). Then:

1. Wrap the main training logic in a `main_worker(rank, world_size)` function.
2. Initialize a distributed process group inside `main_worker` with:
   - `backend="nccl"` (for GPUs) or `"gloo"` (for CPU-only).
   - `init_method="env://"` so you can launch with `torchrun`.
3. Map `rank` to a device (`cuda:{rank}`) and move the model there.
4. Wrap the model in `DistributedDataParallel`.
5. Replace your existing `DataLoader` with one that uses `DistributedSampler`.

Sketch:

```python
def main_worker(rank, world_size):
    # 1. init process group
    # 2. set device based on rank
    # 3. build model, wrap with DDP
    # 4. build dataset + DistributedSampler + DataLoader
    # 5. standard training loop
    ...
```

Launch with:

```bash
torchrun --nproc_per_node=NUM_GPUS train_ddp.py
```

Compare:

- Step time (seconds/iteration) for 1 GPU vs N GPUs.
- Final training/validation metrics after a fixed number of epochs.

Keep the *global* batch size constant when comparing.

### 2. Inspect your sharding: which samples does each rank see?

Instrument your dataset as described earlier:

- Wrap your base dataset to return an `"idx"` field alongside the data.
- In a small debug run (1–2 epochs, tiny dataset), log the indices seen by each rank.

Tasks:

1. Run with `world_size=1` and record the multiset of indices per epoch.
2. Run with `world_size=NUM_GPUS` and gather indices from all ranks.
3. Verify:
   - Every sample appears roughly once per epoch across all ranks.
   - There is no obvious duplication or large gaps.
   - Changing `world_size` doesn’t silently drop a large fraction of the dataset.

Optional: add an assertion that fails if any index count deviates too far from 1 across ranks.

### 3. Profile input pipeline vs compute on 1 GPU and N GPUs

Using your DDP script from Exercise 1:

1. Enable the framework’s profiler (e.g., `torch.profiler`) for a short window (10–20 steps).
2. Capture:
   - CPU time spent in data loading and preprocessing.
   - CUDA time in kernels and collectives (all-reduce).
3. Run twice:
   - `world_size=1`.
   - `world_size=NUM_GPUS` (same global batch size, same model).

Questions to answer:

- Is data loading a larger fraction of step time when using more GPUs?
- Does GPU utilization stay high, or do you see gaps between kernels?
- Does all-reduce time start to become a noticeable part of each step?

Based on what you see, try one small change to your input pipeline (e.g., increase `num_workers`, enable `pin_memory`, or add prefetching) and measure the impact.

### 4. Validate correctness: compare single-GPU vs multi-GPU loss curves

Pick a small configuration that trains in a few minutes. Then:

1. Run a single-GPU baseline with a fixed seed; log loss/accuracy per epoch.
2. Run your DDP version with:
   - Same model and hyperparameters.
   - Same global batch size.
   - Seeded per-rank RNGs (using a helper like `set_seed(base_seed, rank)`).

Overlay the two curves (even roughly, by eye):

- Are they within reasonable variation?
- If not, use the debugging tools from the previous section:
  - Check sharding.
  - Log per-rank losses.
  - Spot-check gradient consistency on a small parameter slice.

The aim is not bitwise identical behavior, but to gain confidence that your distributed run is a faithful, faster version of your baseline—not a different experiment hiding behind more GPUs.

By the end of these exercises, you should have a working mental and practical toolkit for data-parallel training: you can turn a single-device script into a multi-GPU run, verify that data and gradients behave as expected, and reason about performance when you add more hardware.
