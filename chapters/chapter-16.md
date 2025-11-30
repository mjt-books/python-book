# Memory Management, Checkpointing, and Fault Tolerance

This chapter focuses on keeping large, long-running AI workloads alive and efficient: how to fit models into limited device memory, how to save and resume training safely, and how to survive inevitable failures without losing days of progress. You’ll learn practical patterns for managing memory across CPUs, GPUs, and other accelerators, and for making training runs restartable and robust.

Learning objectives:
- Explain common sources of memory pressure in training and inference pipelines.
- Apply practical techniques to reduce peak memory usage on CPUs/GPUs/TPUs.
- Design checkpointing schemes that balance robustness, performance, and storage cost.
- Integrate basic fault tolerance into training loops and orchestration tools.

Sections:
- Why memory and reliability matter at scale
- Anatomy of memory usage in AI workloads
- Practical memory management techniques
- Checkpointing strategies and formats
- Fault tolerance patterns: from single-node to clusters
- Putting it together: resilient training loop patterns
- Exercises: inducing OOMs, recovering from failures, and testing checkpoints

## Why memory and reliability matter at scale

When models are small and experiments are short, you can get away with treating memory and failures as annoyances. Maybe a notebook kernel dies with an out-of-memory (OOM) error once in a while; you drop the batch size, restart, and move on. At scale, the same problems turn into hard limits: you literally cannot fit the model, your cluster spends half its time retrying jobs, and a single bad node can waste thousands of GPU-hours.

Modern AI workloads push every part of the memory hierarchy. Parameters, optimizer states, activations, gradients, input batches, and temporary buffers all compete for limited device RAM. On a single GPU, the difference between “fits comfortably” and “OOMs constantly” might be a small change in sequence length or batch size. In multi-GPU or multi-node setups, you also have to think about host memory, interconnect buffers, and how much you can safely cache on each worker.

Reliability gets harder for similar reasons. Training that used to finish in 30 minutes might now run for days or weeks. Hardware failures, preemptions, software bugs, and cluster maintenance are no longer rare events over that time window—they’re guaranteed. If your only recovery plan is “start again from scratch”, progress becomes fragile and expensive, and experimenting with new ideas feels risky.

The good news is that both memory pressure and failures are predictable in the aggregate. You know that activations scale with batch size, that optimizer choices affect how much state you store, and that large clusters will always have some churn. Instead of hoping for the best, you can design your code and infrastructure around these facts: measure memory, shape it with a few well-chosen techniques, and treat restarts as a normal part of the training loop rather than an emergency.

In practical terms, this chapter will encourage you to think in three complementary dimensions:

- **Capacity:** How much effective memory do you really have across CPUs, GPUs, and storage, and how can you trade compute for memory (e.g., activation checkpointing, offloading) to fit larger models or batches?
- **Continuity:** If the process dies right now, how much work do you lose? Minutes, hours, or days? Checkpointing strategy is what turns random failures into bounded, predictable setbacks.
- **Containment:** When something goes wrong—an OOM, a flaky node, a corrupted file—does it take down the whole run, or can you fail fast, restart safely, and keep the rest of the system healthy?

Later sections will dig into the anatomy of memory usage, concrete management tricks (from mixed precision to sharded optimizers), and fault tolerance patterns for single machines and clusters. The aim is not to make your training immortal, but to make it *boringly robust*: memory usage is measured and intentional, checkpoints are routine, and failures are inconveniences instead of disasters.

## Anatomy of memory usage in AI workloads

Before you can manage memory, you need a rough map of where it actually goes. When people say “my model doesn’t fit on the GPU”, they’re usually talking about more than just parameters. A typical training step has several overlapping consumers of memory:

- **Model parameters** – the weights of your network.
- **Optimizer state** – extra tensors the optimizer keeps (e.g., momentum, Adam’s moments).
- **Activations** – intermediate results stored during the forward pass so backpropagation can reuse them.
- **Gradients** – per-parameter derivatives accumulated during the backward pass.
- **Inputs and batches** – the data you feed the model, often kept in both CPU and GPU memory.
- **Temporary buffers and workspaces** – scratch space used by libraries like cuDNN or cuBLAS.
- **Framework bookkeeping** – graphs, metadata, CUDA streams, and other runtime structures.

On a single device, peak memory usage is often dominated by activations and optimizer state rather than the bare parameters. A transformer that has 10 GB of parameters can easily consume significantly more once you add Adam and a deep stack of activations for long sequences and large batches. This is why “just doubling the batch size” frequently tips you over into OOM territory.

The story gets more interesting when you zoom out to the whole system:

- **Device memory (GPU/TPU/accelerator RAM)** holds parameters, activations, gradients, and kernels’ scratch space. It’s fast and limited; overfilling it leads to OOMs or aggressive eviction.
- **Host memory (CPU RAM)** typically hosts data loaders, preprocessed batches, model checkpoints being written or read, and sometimes offloaded tensors. Poor coordination here can cause the CPU to start swapping, which silently kills performance.
- **Storage (local SSDs, network filesystems, object stores)** stores datasets and checkpoints. Reading too slowly from storage starves GPUs; writing too aggressively can saturate I/O and slow training.

During a single training iteration, memory usage is not flat—there is a **peak**. For many models, this peak happens near the middle of the network, when the largest number of activations are alive. If you’re barely fitting into device memory, that peak is what you need to reduce; shrinking a few small layers or buffers elsewhere won’t help much.

Different phases of the workflow also stress memory in different ways:

- **Training vs inference:** training needs activations and optimizer state; inference can often run with just parameters and a small working set.
- **Forward-only evaluation:** sits somewhere in between—fewer activations than training (no backward), but often run with larger batches.
- **Distributed training:** adds communication buffers, parameter shards, gradient buckets, and sometimes duplicated states on each worker.

The key habit is to stop thinking of “model size” as a single number and instead think in categories: parameters, state, activations, inputs, and overhead, each on CPU, GPU, and storage. Later in this chapter we’ll look at tools to inspect these pieces (framework memory summaries, profilers, OS-level views), and then at techniques that specifically target each category—like activation checkpointing, optimizer sharding, and offloading—to lower the overall peak without rewriting your model from scratch.

## Practical memory management techniques

Once you know what’s using memory, the next step is to shape that usage so your jobs fit comfortably and predictably. In practice you’ll combine a handful of simple, composable techniques rather than rely on a single magic trick. Think of this section as a menu: you won’t need every option for every project, but you should recognize when each one is appropriate.

### Start with measurement and guardrails

Before changing code, add basic visibility:

- Use your framework’s memory reports (e.g., `torch.cuda.memory_summary()`, `jax.profiler`, TensorFlow’s memory stats).
- Track maximum allocated memory per step, not just current usage.
- Add simple assertions where possible: for example, fail fast if a configuration implies a batch size that historically caused OOMs.

On GPUs, get into the habit of checking:

```python
import torch

print(torch.cuda.get_device_name(0))
print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
print("Reserved:", torch.cuda.memory_reserved() / 1e9, "GB")
```

This won’t replace a full profiler, but it lets you see whether a change moved the needle in the right direction.

### Control batch size and sequence length

The most direct levers for peak memory are **batch size** and **sequence/feature length**:

- Halving batch size typically halves activation memory.
- Reducing sequence length or spatial resolution shrinks activations even more aggressively in attention and vision models.

Instead of hard-coding these values deep in your code, surface them as configuration parameters and consider:

- An automatic “find largest safe batch size” mode that binary-searches until training fits.
- Separate batch sizes for training, evaluation, and inference (evaluation can often use larger batches if latency is less critical).

### Use mixed precision and smaller dtypes

Moving from `float32` to `float16` or `bfloat16` can cut memory for parameters, activations, and gradients roughly in half, while often increasing throughput:

- Prefer **AMP** (automatic mixed precision) APIs in PyTorch, TensorFlow, or JAX so you don’t manually sprinkle casts everywhere.
- Keep a master copy of weights in higher precision if your framework recommends it; let the library handle loss scaling.

Be explicit about dtype when creating tensors, rather than relying on defaults:

```python
x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="cuda")
```

For inference, you can often go further with `int8` quantization or even lower, but that usually comes with more tooling and calibration steps (covered in the edge/inference chapters).

### Free what you don’t need, when you don’t need it

Python references keep tensors alive longer than you might expect. A few simple habits help:

- Reuse buffers in hot loops instead of allocating new tensors every step.
- Set large, no-longer-needed tensors to `None` so the garbage collector can reclaim them.
- On GPUs, consider periodic `torch.cuda.empty_cache()` if your allocator tends to hold onto unused blocks for too long (be aware this can increase fragmentation; measure before/after).

When building training loops, avoid storing entire histories of activations, losses, or model outputs unless you truly need them. Logging aggregates (means, percentiles) instead of raw values keeps memory flat over time.

### Activation checkpointing and recomputation

If activations dominate your memory, **activation checkpointing** trades extra compute for lower memory:

- You mark certain layers or blocks so that forward activations are not stored.
- During backward pass, those segments are recomputed.

Frameworks usually provide utilities for this (e.g., `torch.utils.checkpoint`, TensorFlow’s gradient checkpointing, JAX’s `remat`). Good candidates for checkpointing:

- Repeated blocks (transformer layers, residual stacks).
- Deep parts of the network that contribute heavily to peak memory.

Use it selectively: checkpointing every tiny layer can turn backward into an expensive full re-run of the model.

### Shard and offload large states

For very large models, parameters and optimizer state alone can exceed a single device’s memory. Common strategies include:

- **Optimizer sharding / ZeRO-style approaches:** split optimizer state and gradients across multiple GPUs so no single device holds everything. Here, *sharding* means dividing a large tensor or collection of tensors into disjoint pieces and placing different pieces on different devices or processes, rather than keeping full copies everywhere. “ZeRO” (Zero Redundancy Optimizer) is a family of techniques, popularized by DeepSpeed, that partitions optimizer state, gradients, and sometimes parameters across data-parallel workers instead of fully replicating them.
- **Parameter offloading:** keep some parameters or optimizer state on CPU or NVMe instead of GPU, swapping them into GPU memory only when needed. More generally, *offloading* means intentionally storing tensors on a slower but larger memory tier (like host RAM or disk) and moving them to faster memory (like GPU RAM) just-in-time for computation.
- **Activation offloading:** move rarely used or long-lived activations to host memory between forward and backward; you trade extra data transfers and some slowdown for a lower GPU memory peak.

These techniques usually come from libraries (e.g., DeepSpeed, Fully Sharded Data Parallel in PyTorch, distributed optimizers in TensorFlow/JAX). **Fully Sharded Data Parallel (FSDP)** is a PyTorch strategy that shards model parameters, gradients, and optimizer state across GPUs so each rank only holds a slice of the full model at any given time.

The practical guidance:

- Start with built-in sharded or fully sharded data-parallel solutions before building your own.
- Monitor PCIe or network utilization: heavy offloading can simply move your bottleneck from memory to bandwidth.

### Simplify and stabilize the model where practical

Sometimes the cheapest way to “optimize memory” is to pick a slightly smaller or more memory-friendly architecture:

- Use narrower layers or fewer heads when early experiments don’t need full scale.
- Prefer architectures with known memory-friendly implementations (e.g., FlashAttention-based transformers vs naive attention). **FlashAttention** is an optimized attention kernel that computes exact softmax attention using IO-aware tiling, reducing activation memory and improving speed for long sequences.
- Prune unused branches or debugging hooks from your training graph.

This is especially relevant in early research phases: you want to de-risk ideas with a “budget” version of the model, then scale up once the approach is promising and the memory plan is clear.

### Pipeline and prefetch data carefully

Data handling can quietly consume large amounts of host and device memory:

- Cap the number and size of prefetched batches in your dataloaders.
- Avoid keeping multiple full copies of the dataset in RAM unless absolutely necessary.
- When possible, compress data on disk and decompress in streaming fashion rather than eagerly loading everything.

For GPU input pipelines, use pinned memory and asynchronous transfers where supported, but keep an eye on how many in-flight batches you allow—each one is more memory.

---

In the next section we’ll step away from “how to fit the model right now” and look at **checkpointing strategies and formats**: how to turn a working training loop into something you can stop and resume reliably without redoing days of work.

## Checkpointing strategies and formats

Once your job fits into memory, the next question is: *what happens when it dies?* Checkpointing is your main tool for making progress durable. A good checkpointing scheme turns random failures into “lose at most N minutes of work” instead of “start over from the beginning”.

We’ll focus on three aspects:

- **What** to save.
- **When** to save it.
- **How** to store it so you can reliably load it later and on different hardware.

### What to save in a checkpoint

For most training jobs, you want to be able to resume as if the process had never stopped. That usually means saving at least:

- **Model parameters** – the weights of your network.
- **Optimizer state** – momentum buffers, Adam moments, learning rate scheduler state.
- **Training progress metadata** – current epoch, step, global sample count, best validation metric so far.
- **Random number generator (RNG) state** – for Python, NumPy, and your framework (PyTorch/JAX/TF), if you care about reproducibility.

Optional but often useful:

- **Scaler state** for mixed precision (e.g., PyTorch’s `GradScaler`).
- **Data sampler state** – so shuffles or distributed samplers resume in a consistent way.
- **Configuration snapshot** – hyperparameters and model config used for this run.

The rule of thumb: anything that would change training dynamics if it suddenly reset should either be cheap to re-derive or be part of the checkpoint.

### When to checkpoint

Checkpointing too rarely risks losing lots of work; checkpointing too often wastes time and I/O. A common compromise:

- Save **every N steps or M minutes**, whichever comes first.
- Additionally, save a **“best so far”** checkpoint when validation metrics improve.

In long-running jobs, it’s safer to anchor checkpoints to **wall-clock time** (e.g., every 15–30 minutes) rather than epochs, since epoch length can vary with dataset size and data pipeline performance.

Make checkpoint intervals configuration-driven so you can tune them per environment. On preemptible or spot instances, err on the side of more frequent checkpoints; on stable, expensive hardware, you may prefer a slightly longer interval to reduce I/O overhead.

### Where and how to store checkpoints

The storage backend shapes both performance and robustness:

- **Local SSD** – fastest, but ephemeral. Good for intermediate checkpoints, but combine with periodic sync to durable storage.
- **Network filesystem (NFS, Lustre, GCS/S3 via FUSE)** – durable and shared across nodes, but may be slower and more variable.
- **Object storage (S3, GCS, Azure Blob)** – highly durable; typically accessed via SDKs or libraries instead of direct POSIX paths.

Practical patterns:

- Write checkpoints to **a temporary file**, then atomically rename to the final path to avoid partially written files.
- Include **run identifiers, step numbers, and timestamps** in filenames, e.g. `ckpt-step_120000.pt` or `ckpt-epoch_05-val_loss_1.23`.
- Keep a small **rolling window** of the most recent checkpoints (e.g., last 3–5) to bound storage usage while still allowing rollback.

For large models, use formats that support **streaming** or **sharded** checkpoints (e.g., sharded PyTorch state dicts, JAX’s multi-host checkpoints) so that no single process has to hold all parameters and state in memory at once during save/load.

### Designing for load and migration

A checkpoint is only useful if you can load it:

- Treat **checkpoint loading** as part of your normal startup path, not a one-off script. Your training entrypoint should accept “start from scratch” or “resume from this path” via configuration.
- Be tolerant of small changes: if you add or rename layers, write load logic that can skip missing keys or handle mismatches with clear warnings.
- Store **both raw state and a small versioned metadata file** (e.g., a JSON with schema version, framework version, and important hyperparameters). This helps when migrating code or frameworks later.

Cross-hardware concerns:

- If you plan to move between GPU types or to CPU/edge devices, consider also exporting **portable formats** (ONNX, SavedModel, TorchScript) for inference-only uses.
- Separate **training checkpoints** (full state, optimizer included) from **inference artifacts** (weights only, maybe quantized or otherwise transformed).

### Minimizing checkpoint overhead

Checkpointing is I/O-heavy and can briefly spike memory usage:

- Avoid blocking the main training loop for the full duration of a large write. Where possible, offload serialization and upload to a **background thread or process**.
- Compress checkpoints (e.g., using `gzip`, `lz4`, or library-specific compression) if storage or network is the bottleneck. Test the trade-off between CPU time and I/O savings.
- For distributed training, prefer **rank-0-only** checkpointing when feasible, or coordinated sharded checkpoints where each rank writes its shard. Avoid N identical full checkpoints from every process.

---

With these pieces in place, a crash or preemption becomes a routine event: restart the job, point it at the latest checkpoint, and continue. In the next section we’ll broaden the view to **fault tolerance patterns**, looking at how orchestration systems, clusters, and training code all cooperate to detect failures, restart safely, and avoid corrupting checkpoints or clogging your cluster.

## Fault tolerance patterns: from single-node to clusters

When you’re managing memory and checkpointing effectively, adding fault tolerance is mostly about *integration*: making sure that your training loop, checkpointing logic, and orchestration systems (like Kubernetes or SLURM) all work together smoothly.

Key concepts:

- **Health checks** – lightweight probes to detect if a process is alive and responsive.
- **Graceful shutdown** – allowing processes to exit cleanly, finishing current work and saving state.
- **Restart policies** – rules for when and how to restart failed processes or jobs.
- **Job orchestration** – managing the lifecycle and resource allocation of distributed jobs.

### Health checks and monitoring

At scale, you can’t manually inspect every job. Automated health checks are crucial:

- **Process-level checks:**
  - Is the process still running?
  - Is it using CPU/GPU/network as expected?
  - Are there any obvious error messages or logs?
- **Application-level checks:**
  - For a training job, is it making progress (e.g., decreasing loss)?
  - For a serving model, is it responding to requests in a timely manner?

Integrate these checks with your orchestration system’s monitoring tools. Set up alerts for anomalies (e.g., high memory usage, slow response times).

### Graceful shutdown and restarts

Design your training processes to handle termination signals gracefully:

- On receiving a shutdown signal, the process should:
  - Stop accepting new work.
  - Complete the current batch or step.
  - Save a checkpoint if possible.
  - Release resources (e.g., close files, release GPU memory).
- Use **restart policies** in your orchestration system to control how failed jobs are retried:
  - **Immediate restart** for transient errors (e.g., OOM, temporary network issues).
  - **Delayed restart** with exponential backoff for persistent errors.
  - **Fallback** to a different node or resource pool if failures continue.

### Distributed training and fault tolerance

In distributed setups, you have additional considerations:

- **Node failures:** can be detected via timeouts or failed heartbeats. Orchestrators can reschedule tasks on healthy nodes.
- **Task retries:** failed tasks should be retried on the same or different nodes, with careful handling of stateful operations (like writing to a database).
- **Data consistency:** ensure that all nodes have a consistent view of the data, especially after a failure. This might involve reloading data from a reliable source or using versioned data snapshots.

Framework-specific tools can help:

- **PyTorch:** use `torch.distributed.launch` for launching distributed jobs, and handle retries at the application level.
- **TensorFlow:** use `tf.distribute.Strategy` for distribution, and rely on the underlying orchestration system for fault tolerance. `tf.distribute.Strategy` is a high-level TensorFlow API that manages how models and variables are replicated across devices and workers so you can write mostly single-process-style code.
- **Kubernetes:** use Jobs or CronJobs for batch processing, with built-in restart and backoff policies.

---

With these patterns, your training jobs should be able to withstand and recover from a variety of failures, from single-process OOMs to whole-node crashes. In the final section, we’ll put together all the pieces into **resilient training loop patterns**, and cover exercises to test your understanding and skills in inducing OOMs, recovering from failures, and validating checkpoints.

## Putting it together: resilient training loop patterns

So far we’ve treated memory management, checkpointing, and fault tolerance as mostly separate concerns. In real code, they show up together in the same few places: your training loop, your process startup logic, and your shutdown/restart path. This section sketches patterns you can adapt rather than a one-size-fits-all framework.

### A resilient single-process training loop

At minimum, a robust loop should:

- Enforce basic memory guardrails.
- Periodically save checkpoints.
- Handle interrupts (e.g., `SIGTERM`, keyboard interrupt) by exiting cleanly.

Here’s a simplified sketch in PyTorch-style pseudocode:

```python
import signal, time, os
import torch

STOP_REQUESTED = False

def handle_sigterm(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

signal.signal(signal.SIGTERM, handle_sigterm)

def train_loop(model, optimizer, loader, device, cfg):
    model.to(device)
    step = 0
    last_ckpt_time = time.time()

    # Optionally resume
    if cfg.resume_path is not None and os.path.exists(cfg.resume_path):
        state = torch.load(cfg.resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        step = state["step"]

    for epoch in range(cfg.max_epochs):
        for batch in loader:
            if STOP_REQUESTED:
                save_checkpoint(model, optimizer, step, cfg)
                return

            batch = batch.to(device, non_blocking=True)
            # Optional: memory checks or logging
            # assert torch.cuda.memory_allocated(device) < cfg.mem_budget_bytes

            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # Periodic checkpoint by time or steps
            now = time.time()
            if (
                step % cfg.ckpt_every_steps == 0
                or now - last_ckpt_time > cfg.ckpt_every_seconds
            ):
                save_checkpoint(model, optimizer, step, cfg)
                last_ckpt_time = now
```

The important parts are not the exact APIs but the *structure*:

- A single source of truth for `step` and configuration.
- A clear, short path to save state and exit when asked.
- Hooks where you can add memory checks, logging, or profiling without changing the core logic.

### Integrating with distributed and sharded setups

In distributed training, you want similar behavior but with a few twists:

- **One writer:** typically only rank 0 writes “global” checkpoints and metadata; other ranks may write only their parameter/optimizer shards.
- **Synchronized view of progress:** steps/epochs should be consistent across ranks; often derived from a shared global step rather than per-rank counters.
- **Collective shutdown:** when one rank decides to stop (e.g., hits a fatal error), everyone should stop soon after.

A common pattern:

- Wrap your training loop with a small “rank-aware” harness that:
  - Initializes `torch.distributed`/`tf.distribute`/JAX multi-host.
  - Decides, based on rank, whether to perform certain side effects (logging, checkpoint writes).
  - Converts global signals (e.g., `STOP_REQUESTED`) into a broadcast so all ranks exit around the same point.

Checkpoint functions then become thin wrappers over your chosen library’s sharded save/load utilities, plus some rank-0-only metadata handling.

### Making restarts boring

The final piece is to make “start from scratch” and “resume from checkpoint” look almost identical:

- Your main entrypoint should:
  - Parse configuration (including paths, batch sizes, checkpoint cadence).
  - Detect an optional latest checkpoint (e.g., “if `cfg.resume` is `auto`, pick the newest in this directory”).
  - Call the same `train_loop` function with a `resume_path` argument.

Avoid a proliferation of one-off scripts like `train.py`, `resume.py`, `resume_from_spot.py`. Instead, a single, parameterized entrypoint plus a small CLI or config system keeps behavior consistent and testable.

Finally, test this behavior *on purpose*: start a short run, kill it mid-epoch, restart with `resume_path`, and verify loss curves and metrics line up as expected. It’s much easier to debug gaps in your checkpointing or RNG handling with small experiments than in the middle of a multi-day training job.

---

With these patterns in place, you have a coherent story: you know where memory goes, you’ve shaped it so the job fits, you save progress regularly, and you treat failures and restarts as normal operations instead of exceptional disasters. In the exercises, you’ll intentionally trigger OOMs and failures, then practice recovering cleanly and validating that your checkpoints really contain everything you need.

## Exercises: inducing OOMs, recovering from failures, and testing checkpoints

These exercises are designed to make memory limits and failures *visible* and to practice recovering from them in controlled settings. Run them on non-critical machines first—you’re going to crash a few processes on purpose.

### 1. Trigger and analyze a GPU OOM

**Goal:** See how OOMs appear in logs, and how batch size and sequence length affect memory.

1. Start from a simple GPU training script (e.g., a small CNN or transformer).
2. Add a configuration flag for batch size and, if applicable, sequence length or image resolution.
3. Increase these values until you hit a CUDA OOM error.
4. Capture:
   - The exception text and stack trace.
   - The output of `nvidia-smi` (or your accelerator’s equivalent) before and during training.
5. Reduce batch size or sequence length to the largest values that run successfully.

Questions to answer:

- At what configuration did OOMs start?
- How did peak allocated memory (from `torch.cuda.memory_summary()` or similar) change as you scaled up?

### 2. Reduce peak memory with one technique

**Goal:** Apply a single memory optimization and verify that it changes what you measured in Exercise 1.

Using the same model:

1. Pick *one* technique from this chapter, such as:
   - Mixed precision.
   - Activation checkpointing.
   - Smaller batch size + gradient accumulation.
2. Re-run with a configuration that previously OOM’d.
3. Measure:
   - Whether the model now fits.
   - New peak memory usage.
   - Change in step time (did you trade memory for compute?).

Write down a short note comparing “before” and “after” in terms of both memory and speed.

### 3. Implement and test basic checkpointing

**Goal:** Confirm that your training loop can stop and resume without losing significant progress.

1. Add a minimal `save_checkpoint` / `load_checkpoint` pair to your training script that stores:
   - Model parameters.
   - Optimizer state.
   - Current step or epoch.
2. Train for a few epochs, saving a checkpoint every N steps.
3. Manually stop the process mid-run (Ctrl+C or sending a `SIGTERM`).
4. Restart with a `--resume` (or similar) argument pointing to the latest checkpoint.

Verify:

- That training resumes from the expected step/epoch.
- That loss/metric curves are continuous (no huge jump due purely to restart).

### 4. Simulate a preemption or node failure

**Goal:** See how your orchestration or batch system behaves when a job dies, and ensure it can restart cleanly.

Depending on your environment:

- **Single machine:** write a small wrapper script that:
  - Starts your training process in a subprocess.
  - Sends it a `SIGTERM` after a configurable delay.
- **Cluster / Kubernetes / Slurm:** launch a short training job with checkpointing enabled, then:
  - Delete the pod/job or cancel the allocation partway through.
  - Let the system’s restart policy bring it back.

Check:

- Whether a new instance picks up from the latest checkpoint automatically.
- Whether any partial checkpoints or corrupted files were left behind.

### 5. Test checkpoint robustness across code changes

**Goal:** Make sure your checkpoint format and loading code can tolerate small, realistic changes.

1. Train and checkpoint a small model.
2. Make a *non-breaking* change, such as:
   - Adding an extra layer or head.
   - Renaming a module.
   - Adding a new field to your configuration.
3. Try to load the old checkpoint under the new code.

Adjust your load logic (e.g., allowing missing keys, logging warnings) until:

- The model loads without crashing.
- You have clear, intentional behavior for new or missing parameters.

---

By the end of these exercises, you should be familiar with how memory failures manifest, how to relieve memory pressure in at least one concrete way, and how to make training runs restartable and resilient to common failure modes. In later chapters, you’ll apply the same skills to multi-node, multi-accelerator setups where the cost of getting this wrong is much higher.
