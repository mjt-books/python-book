# Multi-node Training and Parameter Servers

This chapter moves from “one big box” thinking to clusters of machines working together. You’ll see how multi-node training actually moves tensors over the network, what changes in your Python code compared to single-node setups, and where parameter servers still make sense versus modern all-reduce–style training.

Learning objectives:
- Explain the main multi-node training patterns (data parallel, model parallel, hybrid).
- Describe how parameter servers and collective all-reduce approaches differ.
- Recognize the roles of networking, topology, and process launchers in distributed runs.
- Set up and run a minimal multi-node training job from Python.

Sections:
- Overview: why and when to go multi-node
- Cluster basics: nodes, networking, and topology
- Multi-node data parallelism with collectives (all-reduce, all-gather)
- Parameter server architectures: concepts and trade-offs
- Fault tolerance, elasticity, and stragglers
- Practical setup: launching and monitoring multi-node jobs
- Exercises: scaling a simple model across multiple nodes

## Overview: why and when to go multi-node

At a high level, multi-node training is just the familiar single-node story repeated across several machines: each process computes gradients, then everyone agrees on updated parameters. The twist is that now your “bottleneck device” is often the **network**, not the GPU. You trade intra-node bandwidth (fast PCIe / NVLink) for inter-node bandwidth (slower, higher-latency links like Ethernet or InfiniBand), and that trade-off reshapes how you design and debug training runs.

There are three main reasons you go multi-node:

1. **Wall-clock time**: you want to finish training much faster by throwing more compute at the problem. That usually means scaling out data parallelism—more workers chewing through more batches in parallel—while keeping validation, logging, and evaluation manageable.

2. **Model or batch size**: a single machine doesn’t have enough memory or compute to fit your model or your preferred batch size. You might distribute layers across nodes (model parallelism), shard optimizer state, or use a hybrid scheme that combines intra-node data parallelism with inter-node model sharding.

3. **Operational constraints**: sometimes the cluster is simply how resources are exposed to you. Cloud-managed training services, on-prem SLURM clusters, or Kubernetes-based ML platforms may encourage or require multi-node setups once you reach certain scales.

The benefits are real—shorter training times, the ability to handle bigger models and datasets—but they come with costs:

- **Communication overhead**: every step requires synchronizing parameters or gradients across nodes. Poorly tuned communication can erase most of your theoretical speedup.
- **Failure modes**: a single flaky node or network hiccup can stall or kill the whole job, especially in tightly-coupled synchronous training.
- **Complexity**: you now care about launchers, hostfiles, environment variables, firewall rules, and timeouts—not just `python train.py`.

From a Python perspective, multi-node training should still feel like “run N copies of my training loop and periodically synchronize state.” The rest of this chapter will unpack how different libraries implement that synchronization (collectives vs parameter servers), what assumptions they make about the cluster, and how to reason about scaling efficiency instead of just “it runs, therefore it must be faster.”

In the following sections, we’ll start with a minimal mental model of a cluster, then build up from pure data-parallel all-reduce to classic parameter-server designs and more elastic schemes. By the end, you should be able to answer a pragmatic question for your own projects: *given my model, dataset, and hardware budget, is multi-node training worth the extra moving parts right now?*

## Cluster basics: nodes, networking, and topology

Before you worry about gradients and optimizers, it helps to picture what the hardware actually looks like at cluster scale. Conceptually, you have:

- **Nodes**: individual machines (physical or virtual) with some CPUs, optional GPUs/TPUs, RAM, and local storage.
- **Network**: links between nodes, usually via one or more network interface cards (NICs).
- **Topology**: how those links are arranged (who is “close” to whom, and over what bandwidth).

From a Python script, these details show up as environment variables, hostnames, and IP addresses, but underneath you are riding on specific network hardware and routing choices that control how fast tensors can move.

Some key ideas:

- **Intra-node vs inter-node**: Within a single node, GPUs might talk over PCIe or NVLink; between nodes, they talk via the NIC. Intra-node bandwidth is usually much higher and latency much lower than inter-node. Good distributed libraries try to use the fastest path available (e.g., NCCL using NVLink inside a node and InfiniBand across nodes).

- **Bandwidth and latency**: Bandwidth is “how many bytes per second” you can move; latency is “how long until the first byte arrives.” Large all-reduce operations are mostly limited by bandwidth; frequent small control messages are more sensitive to latency. Architectures like parameter servers or gradient compression play with this trade-off.

- **Oversubscription and noisy neighbors**: On shared clusters, you rarely own the entire network. Other jobs may be sending traffic across the same links, reducing your effective bandwidth and increasing jitter. This is why the same training job can scale nicely one day and poorly the next, even on identical node types.

You don’t need to become a network engineer, but you should be able to answer a few practical questions about any cluster you use:

- How many GPUs (or accelerators) are there per node?
- What kind of network connects the nodes (1/10/25/100 Gbps Ethernet, InfiniBand, something else)?
- Are there known **islands** (e.g., racks or pods) where intra-island communication is faster than inter-island?

Knowing this lets you choose sensible process layouts. For example, if each node has 8 GPUs, a common pattern is:

- Run 8 worker processes per node (1 per GPU).
- Use an **intra-node** backend (like shared memory or NVLink) to synchronize within the node.
- Use the **inter-node** network for cross-node collectives.

Most modern frameworks hide the low-level transport details, but they still need hints about how many nodes and processes exist, and where each rank runs. That’s where **launchers** and environment variables come in: tools like `torchrun`, `mpirun`, `horovodrun`, `srun`, or Kubernetes job controllers set `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and similar variables so each process knows its place in the cluster.

In the next section, we’ll build on this mental model to look at how multi-node data parallelism actually uses the network: which tensors move, when they move, and how collectives like all-reduce and all-gather turn a pile of independent processes into a coherent distributed optimizer.

## Multi-node data parallelism with collectives (all-reduce, all-gather)

The most common way to scale training across multiple nodes today is **synchronous data parallelism** built on top of **collective communication**. Conceptually, each worker process:

1. Gets its own mini-batch (a shard of the global batch).
2. Runs the forward and backward pass locally.
3. Participates in a collective operation to combine gradients (or parameters).
4. Applies the same optimizer update as every other worker.

If the collectives are correct, all workers keep identical model weights, so the whole setup behaves like training on a single machine with a larger batch.

### What moves over the network?

There are two main patterns:

- **Gradient all-reduce** (most common): each worker computes gradients for all parameters, then everyone participates in an `all_reduce(sum)` on each gradient tensor. After the all-reduce, every worker holds the *sum* of all gradients, which is typically divided by the world size to get the average.

- **Parameter all-gather / broadcast** (less common for dense models): one or a few workers maintain the “authoritative” parameters and broadcast them to others, or workers hold parameter shards and periodically all-gather a full copy. This pattern shows up more in sharded or model-parallel setups.

For standard dense models with dozens or hundreds of MB of parameters, gradient all-reduce is the workhorse. Frameworks like PyTorch, TensorFlow, and Horovod batch these operations to reduce overhead and choose algorithms (ring all-reduce, tree, hierarchical) that map well to your topology.

### All-reduce and friends

A **collective** is an operation that conceptually involves a whole group of processes:

- `all_reduce`: everyone contributes a tensor; everyone receives the reduced result (sum, mean, max, etc.).
- `broadcast`: one process sends a tensor; everyone else receives it.
- `all_gather`: everyone contributes a tensor; everyone receives the concatenation from all ranks.
- `reduce_scatter`: like all-reduce but each process gets only a shard of the final result.

You don’t usually implement these yourself; you call into libraries like NCCL, MPI, Gloo, or one of the higher-level wrappers. The important thing is to know **when** they run in your training loop, because that’s where your step time will spike.

In a typical PyTorch DDP loop:

```python
# Pseudocode, structure only
for batch in loader:
    loss = model(batch)           # forward
    loss.backward()               # backward (computes gradients)
    optimizer.step()              # gradients have already been all-reduced
    optimizer.zero_grad()
```

The `DistributedDataParallel` wrapper hooks into `.backward()` and kicks off asynchronous all-reduce calls as gradients are produced. Your code still looks like single-node training, but every backward pass now implicitly involves network traffic.

### Scaling behavior and bottlenecks

As you add more workers:

- **Compute per worker** goes down (each sees fewer samples per step).
- **Communication volume per step** stays roughly constant per worker (you still need to move gradients for every parameter).

This means your step time tends to become dominated by communication as you scale out. At some point, doubling the number of nodes gives you only a tiny throughput improvement, because all-reduce time dominates compute.

You can often see this in practice by:

- Timing a single training step with 1 node vs 2, 4, 8 nodes.
- Inspecting framework logs or profilers that break out “communication time” vs “compute time”.

If communication is the bottleneck, you can:

- **Reduce gradient size** (mixed precision, gradient compression, sparsification).
- **Overlap communication with computation** (most modern frameworks try to do this automatically).
- **Change batch size per worker** to increase the compute/communication ratio.
- **Use topology-aware algorithms** (hierarchical all-reduce, intra-node first, then inter-node).

### Python-level ergonomics

From Python, the goal is to keep your training loop as close as possible to single-node code and express distributed behavior through:

- A one-time **initialization** call (`init_process_group`, `strategy = ...`, etc.).
- A **wrapper** around your model (`DistributedDataParallel`, `MirroredStrategy`, `Horovod` hooks).
- A **distributed data loader** that shards the dataset by rank.

Example pseudo-structure across frameworks:

```python
def main():
    init_distributed_from_env()   # reads RANK, WORLD_SIZE, etc.

    model = build_model()
    model = wrap_for_distributed(model)  # e.g., DDP, MirroredStrategy

    optimizer = build_optimizer(model)
    loader = build_distributed_dataloader(dataset)

    for batch in loader:
        loss = step(model, optimizer, batch)  # looks like single-node code
```

We’ll look at concrete multi-node examples later in the chapter, but for now the key mental model is:

> Synchronous data parallelism + collectives = many identical workers that periodically agree on shared parameters via all-reduce and related operations.

In the next section, we’ll contrast this collective-based approach with **parameter server architectures**, where a smaller set of nodes owns the parameters and workers send gradients to them instead of talking to each other directly.

## Parameter server architectures: concepts and trade-offs

Before all-reduce–style training became the default, many large-scale systems used (and some still use) **parameter servers (PS)**. The idea is conceptually simple:

- A set of **worker** processes holds data and computes gradients.
- A set of **parameter server** processes owns the model parameters.
- Workers push gradients to the PS, which updates parameters and sends fresh weights back.

Instead of every worker talking to every other worker (as with collectives), workers talk only to the PS tier.

### Basic architecture

In a minimal synchronous setup:

1. Workers fetch the latest parameters (or parameter shards) from the PS.
2. Each worker runs a forward and backward pass on its mini-batch.
3. Workers send gradients to the PS.
4. The PS aggregates gradients, applies the optimizer update, and commits new parameters.
5. Workers repeat with the updated parameters.

Large models are often **sharded** across multiple PS processes (each PS stores a subset of parameters by key or index range), so workers talk to several PS instances per step.

In asynchronous variants:

- Workers do **not** wait for each other.
- Each worker sends gradients as it finishes and pulls parameters whenever it needs them.
- The PS applies updates as they arrive, so different workers may see slightly different versions of the model (so-called **stale gradients**).

### Why parameter servers?

Parameter servers historically appealed to teams building large, heterogeneous systems because they:

- **Fit key/value thinking**: parameters are stored and updated like entries in a distributed key/value store.
- **Allow heterogeneous workers**: slow workers simply contribute fewer updates; they don’t necessarily block the whole system (especially in async mode).
- **Support sharding and specialization**: you can scale PS and worker tiers independently and tune each differently (e.g., more memory on PS nodes, more GPUs on workers).

They also match well with sparse or feature-based models (embedding tables, recommendation systems) where:

- Some parameters are updated very frequently, others very rarely.
- Gradients are sparse; you don’t want to ship full dense tensors around every step.

### Trade-offs vs collectives

Compared with collective-based synchronous data parallelism, parameter servers come with a distinct set of trade-offs:

**Pros**

- **Elasticity and fault tolerance**: it’s often easier to add/remove workers at runtime, or to tolerate stragglers, because workers are not tightly coupled by global barriers.
- **Asynchrony options**: you can choose partially or fully asynchronous updates, which keep fast workers busy even if some nodes are slow.
- **Sparse and sharded updates**: PS architectures can efficiently handle sparse gradients and very large parameter tables by updating only the touched entries.

**Cons**

- **Potentially weaker convergence**: asynchronous updates and stale gradients can hurt convergence or require more careful tuning (learning rates, momentum, etc.).
- **Hotspots and load imbalance**: popular parameters (hot keys) can overload particular PS shards, leading to skewed traffic and bottlenecks.
- **Complexity at scale**: you now manage multiple tiers (workers and PS nodes), routing, sharding, and sometimes consistency models; debugging becomes a distributed systems exercise.

In contrast, all-reduce–based synchronous training:

- Keeps a single, globally consistent view of parameters at each step.
- Tends to be simpler to reason about for dense models.
- Leverages highly optimized collective libraries (NCCL, MPI) that map well to modern GPU clusters.

### Where parameter servers still make sense

Despite the popularity of all-reduce, parameter servers remain relevant in some scenarios:

- **Huge embedding tables** (recommendation, ranking, ads) where model state is much larger than a few GPUs’ memory and is naturally key/value-shaped.
- **Highly heterogeneous clusters** where worker speeds vary a lot and you prefer asynchronous or partially synchronous updates.
- **Online or streaming training** where workers come and go and you want a long-running PS tier that accumulates updates.

From a Python developer’s perspective, PS-based systems usually show up as:

- A **client library** that you use instead of or alongside standard optimizers.
- APIs to **push/pull** parameters or gradients.
- Configuration for **sharding** and routing (often in a YAML or JSON config, not directly in your training loop).

In this chapter, we’ll keep focusing on the mental models rather than a single PS implementation. Later, when we look at concrete frameworks, you’ll see how PS-style designs compare in code to the simpler “wrap your model with DDP and call `torchrun`” approach.

## Fault tolerance, elasticity, and stragglers

Running on multiple nodes means accepting that hardware and networks will fail at some point during training. A GPU might crash, a node might be preempted, or the network might stall long enough for timeouts to fire. When you move from “single big box” to “cluster,” you’re also moving from “it either runs or it doesn’t” to “parts of the system might keep running while others fall over.”

There are three closely related concerns:

- **Fault tolerance**: what happens when a node or process dies?
- **Elasticity**: can the job adapt to changes in cluster size (nodes added/removed)?
- **Stragglers**: how does the job behave when some workers are much slower than others?

### Synchronous collectives and their failure modes

Collective-based synchronous data parallelism assumes that **all ranks participate in every step**. If one worker hangs or dies:

- The others block waiting in an all-reduce or broadcast.
- Timeouts eventually trigger, and most frameworks raise an error and abort the job.
- By default, you lose the current in-memory state unless you’ve been checkpointing.

This “fail-fast” behavior is simple to reason about but unforgiving in long runs, especially on preemptible or shared clusters.

Common mitigation strategies:

- **Frequent checkpoints**: regularly save model weights, optimizer state, and important RNG seeds so you can restart from a recent point instead of from scratch.
- **Shorter jobs with preemption in mind**: structure training as a series of smaller runs that can be resumed, instead of a single multi-day job.
- **Health checks and watchdogs**: monitoring scripts that notice when ranks stop making progress and automatically restart jobs from the latest checkpoint.

Some modern libraries and launchers offer **elastic training** for collectives (e.g., adjusting `WORLD_SIZE` when ranks come and go), but these features are still more constrained and opinionated than classic parameter-server setups.

### Parameter servers, elasticity, and partial progress

Parameter server architectures naturally separate **stateful** components (the PS tier) from more **stateless** or replaceable components (workers):

- If a worker dies, others can often keep training; the PS just sees fewer incoming gradients.
- New workers can be added mid-run and start pulling parameters from the PS.
- The PS tier itself can be replicated and sharded for higher availability.

This makes it easier to build **elastic** training systems where the number of active workers changes over time in response to cluster availability or autoscaling policies.

The trade-off is that you need to think about:

- **Consistency**: how stale can gradients be before convergence suffers?
- **Accounting**: how do you track effective batch sizes and learning rate schedules when worker counts change?
- **PS resilience**: how to avoid the PS tier as a single point of failure (e.g., replication, periodic snapshots).

### Stragglers: slow workers in a fast world

Stragglers are workers that consistently run slower than the rest—because of hardware issues, contention from other jobs, thermal throttling, or simply data skew.

- In **synchronous collectives**, stragglers delay everyone: all-reduce cannot finish until the slowest rank reaches it.
- In **asynchronous PS-style training**, fast workers keep going; slow workers just contribute fewer updates. This can improve utilization but may worsen gradient staleness.

Typical mitigation techniques:

- **Balanced data and workload**: make sure each worker gets similar data sizes and preprocessing cost.
- **Pinning and isolation**: allocate dedicated CPU cores, control NUMA affinity, and isolate I/O-heavy tasks to reduce noisy neighbors.
- **Timeouts and eviction**: detect persistently slow workers and remove them from the job (requires elastic-aware frameworks).
- **Gradient accumulation**: let fewer, faster workers emulate a larger global batch instead of forcing many slow workers to participate.

From a Python point of view, you mostly interact with these ideas through:

- How often you **checkpoint** and what exactly you serialize.
- Which **launcher or strategy** you pick (`torchrun` vs. an “elastic” launcher; synchronous vs. async strategies in higher-level frameworks).
- Simple **monitoring hooks** (logging step times, throughput, loss curves per node) so you can spot stragglers and failures early.

In the next section, we’ll make this more concrete by walking through the **practical setup** of multi-node jobs: how to launch processes, configure environment variables, and monitor runs using common tools like `torchrun`, Slurm, and Kubernetes.

## Practical setup: launching and monitoring multi-node jobs

Abstract mental models are useful, but at some point you need to run `python` on more than one machine. This section focuses on the *plumbing* you touch most often: how to launch distributed jobs, how environment variables wire ranks together, and what to watch while the job runs.

We’ll keep the examples conceptually simple and framework-agnostic, then point to common PyTorch-style commands you can adapt.

### Core ingredients for any multi-node launch

Regardless of framework, you usually have to specify:

- **Number of nodes** (machines) and **processes per node**.
- A **role** for each process (worker, parameter server, coordinator, etc.).
- **Addresses and ports** so processes can find each other.
- A way to pass **rank information** into each process.

Most tools translate this into a small set of environment variables, e.g.:

- `WORLD_SIZE`: total number of processes participating.
- `RANK`: unique ID of this process, from `0` to `WORLD_SIZE - 1`.
- `LOCAL_RANK`: ID of this process on its node (e.g., `0..7` for 8 GPUs).
- `MASTER_ADDR`, `MASTER_PORT`: address/port of a rendezvous server or rank 0.

Your Python entrypoint then does something like:

```python
def main():
    # Pseudocode — each framework has its own init call
    init_distributed_from_env()

    # Use LOCAL_RANK to pick the GPU on this node
    device = pick_device_from_env()

    model = build_model().to(device)
    model = wrap_for_distributed(model)

    # Datasets are usually sharded by global rank
    loader = build_distributed_dataloader(rank=get_rank(), world_size=get_world_size())

    train(model, loader, device)
```

The launcher’s job is to set these environment variables correctly before starting `main()` on each node.

### Example: torchrun on two nodes

On a simple 2-node cluster where each node has 4 GPUs, a minimal PyTorch-style launch might look like:

On **node 0**:

```bash
MASTER_ADDR=node0.example.com
MASTER_PORT=29500

torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train_multinode.py
```

On **node 1**:

```bash
MASTER_ADDR=node0.example.com
MASTER_PORT=29500

torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=1 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train_multinode.py
```

`torchrun` will:

- Start 4 processes per node (one per GPU).
- Set `WORLD_SIZE=8`, `RANK` and `LOCAL_RANK` appropriately.
- Call into your script, where `init_process_group` (or equivalent) reads these variables.

You don’t have to use PyTorch to follow this pattern; many frameworks and custom launchers follow the same basic contract.

### Example: Slurm-style launch

On clusters that use Slurm, you often let the scheduler decide node allocation and then launch one process per GPU:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=mnist-multinode

module load cuda            # cluster-specific
source venv/bin/activate    # your env

srun python train_multinode.py
```

Many frameworks detect Slurm variables like `SLURM_PROCID` and `SLURM_NTASKS` and convert them to `RANK`/`WORLD_SIZE` inside the script, or you can do it yourself:

```python
import os

rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NTASKS"])
local_rank = int(os.environ.get("SLURM_LOCALID", 0))
```

Kubernetes operators and cloud training services follow analogous patterns: they inject environment variables or config files describing the cluster, and your code calls an “init from env” helper before doing anything else.

### Quick checklist for a first multi-node run

When moving from single-node to multi-node, it’s worth verifying a few basics before debugging performance:

1. **Can nodes see each other?**  
   - Test simple `ping`/`ssh` between nodes.
   - Ensure firewalls allow traffic on the chosen `MASTER_PORT`.

2. **Is the environment consistent?**  
   - Same Python version, library versions, and CUDA/NCCL builds on all nodes.
   - Same code and configuration deployed to each node.

3. **Does a minimal distributed script run?**  
   Start with something that only initializes the distributed backend and runs an all-reduce on a tiny tensor. If that fails, fix it before adding real models.

4. **Are ranks mapped to devices correctly?**  
   - Log `RANK`, `LOCAL_RANK`, and `torch.cuda.current_device()` (or equivalent) at startup.
   - Make sure each GPU is used by exactly one process.

### Monitoring multi-node jobs

Once the job starts, you’ll want visibility on three levels:

- **Node-level health**: CPU, memory, GPU utilization, and network I/O.
  - Tools: `nvidia-smi -l`, `htop`, `iftop`, cluster dashboards.
- **Process-level logs**: per-rank loss curves, step times, and communication errors.
  - Prefix logs with rank IDs (e.g., `[rank 3] loss=...`) to distinguish them.
- **Framework-level metrics**: built-in profilers or hooks that expose communication time, GPU utilization, and data-loader performance.

Simple but effective patterns:

- Have rank 0 log global metrics (loss, throughput) and save checkpoints.
- Have non-zero ranks log more sparingly, focusing on debug info like step durations.
- When something stalls, check:
  - `nvidia-smi` for one or more GPUs stuck at 0% utilization.
  - Logs for timeouts or “hang in all_reduce” messages.
  - Network stats for saturated or flapping links.

In the next section, we’ll wrap up the chapter with **exercises**: you’ll take a small single-node training script and scale it to multiple nodes, then measure how close your speedup gets to the ideal and how communication shows up in your profiles.

## Exercises: scaling a simple model across multiple nodes

These exercises are meant to turn the ideas in this chapter into concrete, measurable experiments. You don’t need a huge cluster—two small GPU nodes or even two CPU-only VMs are enough to see the patterns.

### 1. Turn a single-node script into a 2-node job

Take a small training script you already have (or a tutorial model) and:

1. Refactor it into a `main()` that:
   - Reads `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` from the environment.
   - Picks a device based on `LOCAL_RANK`.
   - Wraps the model in your framework’s distributed abstraction (e.g., DDP).
   - Uses a distributed/sharded data loader.

2. Run it on a single node with `nproc_per_node > 1` (multi-GPU or multi-process on CPU).
3. Then run it across 2 nodes using a launcher like `torchrun` or your cluster’s equivalent.

Record:

- Effective samples/second (or steps/second) in each configuration.
- Any changes you had to make to logging, seeding, or checkpointing to keep runs reproducible.

### 2. Measure scaling and communication overhead

Using the same script:

1. Run with `1, 2, 4, 8, ...` workers (processes) and measure:
   - Step time (average over e.g. 100 steps).
   - Throughput (samples/second).

2. Plot or tabulate:

- Number of workers vs. throughput.
- Number of workers vs. step time.

3. Use your framework’s profiler or logs to estimate how much of each step is spent in:

- Forward/backward compute.
- Communication (all-reduce, all-gather, etc.).
- Data loading.

Questions to answer:

- At what point does adding more workers stop giving near-linear speedups?
- How does communication time grow as you add nodes?

### 3. Introduce an artificial straggler

Pick one worker and make it slower, for example by:

- Sleeping for a small random amount of time in its training loop, or
- Giving it a heavier data augmentation pipeline.

Observe:

- In a synchronous all-reduce setup, how much does overall step time increase?
- Does the slower worker become obvious in logs or monitoring dashboards?
- How would you detect and mitigate similar stragglers in a real job?

If you have access to a PS-style or elastic training setup, repeat the experiment and compare how stragglers affect throughput and convergence.

### 4. Experiment with batch size and gradient accumulation

Using the same hardware:

1. Fix the **global batch size** and vary:

   - Number of workers.
   - Per-worker batch size (using gradient accumulation to keep the global batch constant).

2. Compare:

- Convergence behavior (loss curves).
- Step time and throughput.
- Communication/computation ratio.

Try to answer:

- For your model and hardware, is it better to:
  - Use more workers with smaller batches each, or
  - Use fewer workers with larger per-worker batches?

### 5. Optional: try a different topology or network

If you have access to different environments (e.g., on-prem InfiniBand vs. cloud VMs over Ethernet):

- Run the same multi-node experiment in each environment.
- Keep model, code, and configuration identical.
- Compare:
  - Step times and scaling.
  - Reported bandwidth/latency from profiling tools.

Write down a short note summarizing what you observed:

- Where did your job stop scaling well?
- How visible were network effects?
- Given these results, would you choose multi-node training for this model and dataset in your day-to-day work?

These small experiments will give you a grounded sense of how multi-node training behaves on *your* hardware, making the patterns in later chapters (multi-node GPU clusters, TPUs, Kubernetes, and parameter-server–style systems) easier to evaluate and adapt.
