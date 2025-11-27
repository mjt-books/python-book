# Ray, Dask, and Distributed Compute Frameworks for Python

This chapter introduces Python-first distributed compute frameworks and shows how they help you scale beyond a single machine or a single GPU. Instead of hand-rolling process management, RPC, and scheduling, you’ll learn how to use Ray, Dask, and similar tools as the “control plane” for clusters of CPUs and GPUs.

We’ll focus on practical patterns you can drop into existing code: parallelizing data preprocessing, fanning out hyperparameter searches, and coordinating multi-GPU or multi-node training jobs. Along the way, you’ll see how these frameworks interact with lower-level libraries (PyTorch, TensorFlow, JAX, cuDF, etc.) and what trade-offs they make in latency, throughput, and operational complexity.

Learning objectives:
- Explain when to reach for Ray, Dask, or similar frameworks instead of manual multiprocessing or MPI.
- Describe the core abstractions each framework offers (tasks, actors, futures, distributed collections).
- Understand how these frameworks schedule work across CPUs, GPUs, and nodes.
- Identify common patterns for scaling training, inference, and preprocessing pipelines.
- Recognize operational considerations: monitoring, fault tolerance, and resource management.

Sections:
- Why distributed frameworks on top of Python?
- Core abstractions: tasks, actors, and distributed collections
- Ray in practice: tasks, actors, and Ray Train
- Dask in practice: arrays, dataframes, and custom graphs
- Integrating with deep learning frameworks and GPUs
- Operational concerns: clusters, schedulers, and observability
- Exercises: small-scale distributed experiments on your laptop

## Why distributed frameworks on top of Python?

If you’ve ever wrapped a `for` loop in `multiprocessing.Pool`, launched a few training jobs by hand on different machines, or written a quick bash script to fan out experiments, you’ve already felt the need for distributed frameworks. Those ad‑hoc solutions work for a while, but they break down as soon as you have more data, more models, or more hardware than one person can comfortably juggle.

The core problem is coordination: deciding *what* runs *where* and *when*, and doing it reliably. Plain Python gives you threads and processes, but once you cross machine boundaries you’re suddenly dealing with SSH, queues, RPC, failures, retries, and resource contention. You can build all of this yourself, but you’ll essentially be writing a small, bespoke cluster scheduler every time.

Ray, Dask, and similar frameworks exist to standardize that control plane. Instead of manually managing processes and machines, you describe your work in familiar Python terms—functions, objects, and collections—and let the framework:

- Launch and tear down workers across CPUs, GPUs, and nodes.
- Move data where it needs to be, as lazily as possible.
- Track dependencies between tasks so they run in a safe order.
- Recover from worker failures without you restarting everything.

This matters for AI workloads because “just add more hardware” is only useful if you can *keep it busy*. Training a single large model, preprocessing terabytes of data, running thousands of hyperparameter trials, or serving many inference requests per second all benefit from a scheduler that can see the whole cluster and make sensible decisions.

Distributed frameworks on top of Python also respect the reality that most AI code is written in Python already. Instead of rewriting your pipelines in a DSL or low‑level system, you typically:

- Keep your core model and preprocessing code as regular Python functions.
- Use Ray or Dask as a thin layer that parallelizes and distributes those functions.
- Plug into existing libraries (PyTorch, TensorFlow, scikit‑learn, pandas, cuDF) rather than replacing them.

The rest of this chapter will treat these frameworks as another kind of hardware abstraction. Just as earlier chapters showed Python steering CPUs and GPUs, here you’ll see Python steering *clusters*—with Ray, Dask, and friends acting as the bridge between your code and many machines.

## Core abstractions: tasks, actors, and distributed collections

Underneath the marketing pages and cluster diagrams, most Python distributed frameworks are built from a small set of ideas:

- A **task** is a function call that can run somewhere else.
- An **actor** is a stateful object whose methods can run somewhere else.
- A **future** (or “remote handle”) is a placeholder for a result that hasn’t finished yet.
- A **distributed collection** is a big array or table that’s been split into many chunks and spread across workers.

Everything else—hyperparameter tuning, data pipelines, distributed training—is composed from these pieces.

In Ray, you’ll decorate a function with `@ray.remote` and call it with `.remote(...)` instead of `(...)`. That tiny change tells Ray, “this is a task; schedule it wherever you like.” The return value is a future, not the actual result. You can launch thousands of such tasks quickly, then wait for them in batches as results arrive. For long-lived, stateful components (a replay buffer, a model shard, a service), Ray gives you actors: Python classes decorated with `@ray.remote` whose instances live on the cluster.

Dask takes a more data-centric approach. Its core abstractions are distributed collections—`dask.array`, `dask.dataframe`, and `dask.bag`. Each one looks like its in-memory counterpart (NumPy array, pandas DataFrame, list of Python objects) but is actually a lazily built computation graph over many small chunks. When you call `.compute()`, Dask turns that graph into tasks, schedules them across workers, and stitches the pieces back together.

The **future** concept appears in both worlds. Ray futures are explicit: you see them as `ObjectRef` values and use `ray.get()` or `ray.wait()` to retrieve or await them. Dask futures can be exposed through its “distributed” API as well, letting you submit arbitrary Python functions and hold on to references without materializing the results immediately. In both cases, futures let you overlap compute and communication naturally: while some tasks are still running, you can already launch new ones, inspect partial results, or cancel work.

From a mental-model perspective, these abstractions mirror patterns you already know from single-machine Python:

- A task is like calling a function in an `asyncio` event loop, but the event loop is now a cluster scheduler.
- An actor is like an object living in another process, with methods invoked via RPC instead of direct calls.
- A distributed collection is like a big NumPy or pandas object that happens to be sharded across machines.

The key difference is that the framework tracks *dependencies* and *data locality* for you. When you write `z = x + y` on two distributed arrays, it figures out which chunks of `x` and `y` live where, creates the right tasks, and moves data as little as possible. When you call an actor method, it routes the call to the correct worker and serializes only what’s needed.

Throughout the rest of the chapter, we’ll keep referring back to these building blocks. Ray’s higher-level libraries (like Ray Train and Ray Tune) and Dask-based systems (like distributed ETL or feature engineering pipelines) are “just” larger blueprints made from tasks, actors, futures, and collections. If you’re comfortable with these primitives, you can understand what higher-level APIs are really doing—and drop down a level when you need more control.

## Ray in practice: tasks, actors, and Ray Train

Ray’s APIs are deliberately small: you get tasks, actors, object references, and a runtime that understands resources (CPUs, GPUs, custom accelerators). The power comes from how easily you can plug those pieces into existing Python code.

At the simplest level, you start a Ray runtime and mark functions as remote:

```python
import ray

ray.init()  # or ray.init(address="auto") to connect to a cluster

@ray.remote
def preprocess_batch(batch):
    # ... CPU-heavy or GPU-heavy work ...
    return transformed_batch

# Launch tasks in parallel
futures = [preprocess_batch.remote(b) for b in batches]

# Block only when you need the results
results = ray.get(futures)
```

The only difference from a normal Python function is the `@ray.remote` decorator and the `.remote(...)` call. Ray turns each call into a task, schedules it somewhere in the cluster, and returns an `ObjectRef` future. You can create thousands of tasks quickly; Ray will queue and execute them as resources free up.

For stateful components, you switch to actors:

```python
@ray.remote
class EmbeddingCache:
    def __init__(self):
        self._cache = {}

    def get_or_compute(self, key, compute_fn):
        if key not in self._cache:
            self._cache[key] = compute_fn(key)
        return self._cache[key]

cache = EmbeddingCache.remote()

# Methods are invoked remotely; results come back as futures
emb_ref = cache.get_or_compute.remote("item_123", compute_fn)
emb = ray.get(emb_ref)
```

Here, `EmbeddingCache` lives on a worker in the cluster, but you interact with it as if it were a local object. Ray handles placement, method routing, and serialization.

Ray’s resource model lets you express what each task or actor needs. For example, a training step that should run on 1 GPU and 2 CPU cores can be declared like this:

```python
@ray.remote(num_cpus=2, num_gpus=1)
def train_step(config, shard):
    # Inside, you write normal framework code:
    # select the visible GPU, build the model, run a few epochs, report metrics, ...
    return metrics
```

When you launch many `train_step.remote(...)` calls, Ray will only schedule them on nodes where that combination of resources is available, automatically packing and spreading work across your cluster.

On top of these primitives, Ray provides higher-level libraries tailored for ML:

- **Ray Train** focuses on distributed training. It gives you strategies for data parallelism (multiple workers each training a copy of the model) and integrates with frameworks like PyTorch, TensorFlow, and XGBoost. You write a single-worker training function and let Ray Train handle the distributed launch, communication setup, and checkpoint coordination.
- **Ray Tune** handles hyperparameter search, trial scheduling, and early stopping. Under the hood, each trial is just a Ray task or actor running your training function with a different configuration, but Tune manages the bookkeeping.

A typical pattern is:

1. Write a single-node training function that takes a configuration and (optionally) a data shard.
2. Wrap it with Ray Train or invoke it from Ray tasks.
3. Use Ray’s dashboard and logs to monitor throughput, GPU utilization, and failures across the cluster.

The important point is that you don’t have to start with a fully distributed design. You can begin by making one or two functions remote, verify that they behave correctly on your laptop (Ray can run in “local mode” too), and only then point the same code at a multi-node cluster. The abstractions from the previous section—tasks, actors, futures—stay the same; you’re just giving Ray more hardware to work with.

## Dask in practice: arrays, dataframes, and custom graphs

Where Ray starts with tasks and actors, Dask starts with *collections* and a *graph*. You write code that looks like NumPy or pandas, and Dask quietly builds a task graph representing all the work needed to produce the final result. Only when you call `.compute()` (or similar) does it hand that graph to a scheduler and execute it across threads, processes, or a cluster.

A typical pattern with `dask.array` looks like this:

```python
import dask.array as da

# Create a large array split into 100 MiB chunks
x = da.random.random((100_000, 10_000), chunks=(10_000, 1_000))

# Build a computation graph: normalize columns, compute column-wise means
x_norm = (x - x.mean(axis=0)) / x.std(axis=0)
col_means = x_norm.mean(axis=0)

# Nothing has executed yet.
# Trigger execution and pull the result into memory:
result = col_means.compute()
```

Each Dask array is a grid of smaller NumPy arrays (“chunks”). Operations like `mean` or `+` are implemented as many small tasks over chunks, plus some reduction and combination steps. Dask keeps track of dependencies between these tasks so it can parallelize safely and reuse intermediate results when possible.

For tabular data, `dask.dataframe` mirrors a subset of the pandas API:

```python
import dask.dataframe as dd

# Lazily read many parquet files into a partitioned Dask DataFrame
df = dd.read_parquet("s3://bucket/dataset/*.parquet")

# Filter, feature-engineer, and groupby using (mostly) familiar pandas syntax
filtered = df[df["country"] == "US"]
features = filtered.assign(log_price=dd.log(filtered["price"] + 1))

by_category = features.groupby("category")["log_price"].mean()

# Trigger the actual work and bring a small result back to the driver
summary = by_category.compute()
print(summary.head())
```

Under the hood, Dask represents this as a directed acyclic graph (DAG) of tasks. Each partition corresponds to a chunk of your dataset; operations become graph nodes; edges encode dependencies. The default threaded or multiprocessing scheduler can execute this graph on a single machine. With `dask.distributed`, you can push the same graph to a remote cluster:

```python
from dask.distributed import Client

# Connect to an existing Dask scheduler (local or remote)
client = Client("tcp://scheduler:8786")

# Now .compute() runs on the cluster rather than just your laptop
summary = by_category.compute()
```

For situations where the built-in collections are too constraining, you can build **custom graphs** directly using `dask.delayed` or the distributed client’s `submit` API:

```python
from dask import delayed

@delayed
def load_shard(path):
    # ... read file, parse, return a NumPy array or pandas DataFrame ...
    return shard

@delayed
def process_shard(shard):
    # ... feature engineering, filtering, etc. ...
    return processed

@delayed
def combine(shards):
    # ... concatenate, aggregate, or write out ...
    return combined

paths = ["shard-000.parquet", "shard-001.parquet", "shard-002.parquet"]
shards = [process_shard(load_shard(p)) for p in paths]
result = combine(shards)

# Build the full graph and run it
final_value = result.compute()
```

This style feels more like writing a small DAG by hand, but you still get automatic parallelism, fault handling, and scheduling. It’s useful when your workflow doesn’t map neatly to “big array” or “big dataframe” abstractions, or when you want tight control over phase boundaries (e.g., “load and preprocess,” then “train,” then “write metrics”).

In AI pipelines, Dask often shines in the *data* side of the house:

- Converting raw logs or parquet into training-ready features.
- Joining large tables of user/item metadata.
- Computing aggregate statistics over many partitions of data.
- Feeding GPUs with preprocessed batches via iterators or writers.

You can think of it as the “data parallel” partner to your training framework. Your model code stays in PyTorch, TensorFlow, or JAX; Dask focuses on getting bytes off disk, through transformations, and into a shape your trainers can consume—without you having to hand-roll yet another data-prep cluster script.

Later in this chapter, we’ll return to how Dask-based preprocessing and Ray- or framework-based training can fit together into a larger, end-to-end pipeline.

## Integrating with deep learning frameworks and GPUs

So far we’ve treated Ray and Dask as generic distributed engines. In practice, you’ll almost always use them *with* a deep learning framework—PyTorch, TensorFlow, JAX—and *on* GPU-equipped machines. The key is to let each layer do the job it’s best at:

- Ray/Dask: scheduling work across processes, nodes, and devices.
- PyTorch/TF/JAX: expressing and executing tensor computations on specific devices.
- CUDA/cuDNN/cuBLAS/etc.: implementing fast kernels.

From your Python code, this usually looks like:

1. Use Ray or Dask to fan out *units of training or inference work* (e.g., “train on this shard with this config”).
2. Inside each unit, write normal framework code that selects a device, builds a model, and runs a loop.
3. Be explicit about GPU allocation so workers don’t fight over the same device.

### Ray + PyTorch: one process per GPU

The most common pattern with Ray is “one worker process per GPU.” You declare the GPU requirement at the task or actor level, and then let your training code assume `cuda:0` inside the process:

```python
import ray
import torch

ray.init(address="auto")  # or local ray.init() for testing

@ray.remote(num_gpus=1)
def train_on_shard(config, shard_path):
    # Each worker sees a single logical GPU as cuda:0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    loader = make_dataloader(shard_path, batch_size=config["batch_size"], device=device)

    for epoch in range(config["epochs"]):
        for batch in loader:
            # assume batch is already on the right device
            out = model(batch["inputs"])
            loss = compute_loss(out, batch["targets"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return collect_metrics(model)
```

The important division of responsibilities:

- Ray decides *which* GPU and node to use (`num_gpus=1`).
- PyTorch only needs to know “there is a GPU here” (`cuda:0`).

Ray Train formalizes this pattern and adds built-in strategies like fully sharded data parallel (FSDP), distributed data parallel (DDP), and checkpoint broadcasting, but the underlying idea is identical: Ray launches N workers with GPUs, your training function is written as if it were single-node, and the library wires up the process group.

### Dask for data, framework for training

With Dask, a common split is:

- Use Dask to prepare and shard large datasets.
- Export those shards to a storage location or hand them off via an iterator.
- Let your training framework (optionally coordinated by Ray or another tool) handle GPU-side work.

For example, you might:

```python
import dask.dataframe as dd

# 1. Preprocess and shard a large dataset with Dask
df = dd.read_parquet("s3://raw-bucket/logs/*.parquet")

processed = (
    df[df["country"] == "US"]
    .assign(
        log_price=dd.log(df["price"] + 1),
        # ...more features...
    )
)

# Write out many small shards that GPU trainers can read in parallel
processed.repartition(npartitions=128).to_parquet("s3://clean-bucket/train_shards/")
```

Then, separately, you use PyTorch + Ray to consume those shards:

```python
import glob
import ray

shards = glob.glob("s3://clean-bucket/train_shards/*.parquet")

@ray.remote(num_gpus=1)
def train_on_shard(shard_path):
    # ... PyTorch training code that reads this shard and trains on cuda:0 ...
    return metrics
```

This keeps concerns separated:

- Dask focuses on high-throughput IO and transformation over many CPU workers.
- Trainers focus on keeping GPUs busy with well-structured batches.

If you prefer everything in one system, you can also run Dask and your training framework in the same cluster (for example, using `dask_cuda` workers on GPU machines and pushing some preprocessing to the GPU with RAPIDS/cuDF). The trade-off is more moving parts; a clean “data job, then training job” pipeline is often easier to reason about.

### Practical tips for GPU integrations

A few habits help avoid subtle bugs and underutilized hardware:

- **Pin one logical GPU per process** whenever possible. Ray’s `num_gpus=1` and tools like `CUDA_VISIBLE_DEVICES` simplify device logic inside your training function.
- **Avoid large host–device ping-pong** in distributed setups. Do as much preprocessing as possible on the same side (CPU or GPU) where it will be consumed.
- **Warm up kernels** before timing or profiling; framework runtimes and CUDA JIT compilation can make the first few steps unrepresentative.
- **Monitor utilization** with tools like `nvidia-smi`, Ray’s dashboard, and Dask’s dashboard to confirm that CPUs, GPUs, and network are all doing useful work.

In the next section, we’ll zoom out from individual jobs and look at how to run these frameworks in real clusters, including Kubernetes and Slurm-style environments, and how to keep an eye on scheduling, failures, and observability.

## Operational concerns: clusters, schedulers, and observability

Once you move beyond a single machine, the hard problems stop being “can my code launch a task?” and start being “can my jobs survive real infrastructure?” Clusters introduce queueing, failures, upgrades, noisy neighbors, and resource quotas. Ray and Dask can help, but they sit on top of a *cluster substrate* that you still need to understand at a high level.

Most teams end up in one of three setups:

- **Bare-metal or simple VM clusters**, where you start Ray or Dask directly on machines.
- **Scheduler-managed clusters** (Slurm, PBS, etc.), where jobs request nodes and run for a fixed time.
- **Container orchestration** (usually Kubernetes), where pods are scheduled onto nodes and a controller manages Ray/Dask clusters.

The good news is that, from your Python code, these all look similar: you connect to a scheduler endpoint and submit work. The differences are in how clusters are *created*, how long they live, and how you observe what’s going on.

### Cluster lifecycles and schedulers

With **bare-metal or static VMs**, you typically:

1. Provision N machines.
2. Start a Ray/Dask scheduler on one of them.
3. Start Ray/Dask workers on the others, pointing at the scheduler.
4. Run many jobs against this long-lived cluster.

This is simple to reason about and great for interactive development, but it can be wasteful if machines sit idle.

With **Slurm-style schedulers**, your job is more ephemeral:

1. You submit a batch job requesting resources (e.g., `#SBATCH --nodes=4 --gres=gpu:4`).
2. Slurm allocates nodes, and your job script runs on them.
3. Inside that script, you start a Ray or Dask cluster (scheduler + workers) on the allocated nodes.
4. Your Python driver script connects to `ray.init(address="auto")` or a known Dask scheduler address, does its work, and exits.
5. When the batch job ends, the cluster disappears.

This pattern keeps clusters tightly scoped to the job, which simplifies cleanup and cost accounting.

On **Kubernetes**, a common pattern is:

- Deploy a Ray or Dask *cluster CRD* (using the official operators) that defines how many workers and which resource types (CPU, GPU) you want.
- Let Kubernetes create pods for the scheduler and workers.
- Run your Python driver either inside the cluster (as another pod) or externally, connecting to the scheduler’s service.

Scaling becomes a matter of editing the cluster spec (or using autoscaling hooks) instead of manually starting/stopping daemons.

### Resource management and quotas

Whatever substrate you use, you’ll need to think about:

- **CPU/GPU quotas**: how many cores and GPUs your job is allowed to use.
- **Memory limits**: per-process and per-node memory caps.
- **Job timeouts**: maximum wall-clock time in batch systems.

Ray and Dask expose resource-awareness through their APIs:

- In Ray, `num_cpus`, `num_gpus`, and custom resource tags (`resources={"tpu": 1}`) control placement.
- In Dask, worker configuration (`--nthreads`, `--nprocs`, memory limits) and resource tags define what each worker can run.

Align these with your scheduler’s view of the world. For example, if Slurm gives you a node with 8 GPUs and you start 8 Ray workers each with `num_gpus=1`, your tasks can safely assume “one GPU per process.” If you misconfigure this and oversubscribe GPUs, you’ll see unpredictable slowdowns and out-of-memory errors that are hard to debug.

### Observability: dashboards, logs, and metrics

Distributed systems are opaque unless you invest in visibility. Both Ray and Dask ship with useful tooling:

- **Ray dashboard**: a web UI showing cluster resources, task timelines, actor lifecycles, logs, and per-node health. When you run `ray.init()`, it prints the dashboard URL; on clusters, operators usually expose this via an HTTP endpoint.
- **Dask dashboard**: a Bokeh-based UI with task streams, worker utilization, memory usage, and scheduler state. It’s typically accessible at the scheduler’s address (e.g., `http://scheduler:8787`).

You should get into the habit of:

- Watching the **task stream** to see if work is evenly spread or bottlenecked.
- Checking **worker memory** to spot leaks or skewed datasets.
- Looking at **failed tasks** and exceptions instead of only reading the terminal of your driver process.

For production or CI systems, you’ll often integrate with broader observability stacks:

- Export Ray/Dask metrics to **Prometheus + Grafana** for long-term dashboards.
- Ship worker logs to centralized log systems (ELK, Loki, cloud logging) with tags for job ID, node, and worker type.
- Set up basic alerts on “no workers connected,” “high failure rate,” or “scheduler unresponsive.”

### Failure modes and fault tolerance

Finally, real clusters fail in real ways:

- Nodes disappear (preemptible instances, hardware failures).
- Network partitions delay or drop heartbeats.
- Workers hit OOM and get killed by the OS or container runtime.

Ray and Dask can retry tasks and reschedule work on surviving nodes, but only if:

- Your code is **idempotent** (it’s okay to run a task twice).
- State is either externalized (e.g., in object stores, databases) or managed via robust patterns (e.g., Ray actors with checkpointing).
- You avoid hidden side effects in tasks that would corrupt downstream work on re-execution.

When you design jobs, prefer many small, restartable tasks over a few giant, fragile ones. Combine that with good observability, and you’ll spend more time improving models and less time chasing mysterious “the cluster was fine but nothing finished” incidents.

In the final section of this chapter, we’ll shift back to a more hands-on mode with small exercises you can run on your laptop: spinning up mini Ray and Dask clusters, submitting toy jobs, and using the dashboards to see what your code is actually doing.

## Exercises: small-scale distributed experiments on your laptop

These exercises mirror the “quick setup” style from chapter 1, but now focused on Ray and Dask. The goal is to build intuition for how tasks, actors, and collections behave—without needing a real cluster. All of them should run on a single laptop; if you have more cores, they’ll simply run faster.

### 1. Ray: fan out CPU-bound work

Create a script that estimates π using many independent trials, first in plain Python and then with Ray tasks:

```python
import math
import random
import time

import ray

def estimate_pi_serial(num_samples: int) -> float:
    inside = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / num_samples

if __name__ == "__main__":
    N = 1_000_000

    t0 = time.perf_counter()
    pi_serial = estimate_pi_serial(N)
    t1 = time.perf_counter()
    print(f"Serial π estimate: {pi_serial:.6f} in {t1 - t0:.2f}s")

    ray.init()

    @ray.remote
    def estimate_pi_chunk(num_samples: int) -> float:
        return estimate_pi_serial(num_samples)

    chunks = 8
    futures = [estimate_pi_chunk.remote(N // chunks) for _ in range(chunks)]

    t0 = time.perf_counter()
    partials = ray.get(futures)
    pi_parallel = sum(partials) / len(partials)
    t1 = time.perf_counter()
    print(f"Ray π estimate:    {pi_parallel:.6f} in {t1 - t0:.2f}s")
```

Questions to explore:

- How does speedup change as you vary `chunks`?
- What happens if you set `num_cpus=1` on the remote function vs leaving it implicit?
- Open the Ray dashboard and watch the tasks run.

### 2. Ray: simple actor with shared state

Extend the previous script to keep a shared counter in an actor:

```python
import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.n = 0

    def add(self, x: int):
        self.n += x

    def value(self) -> int:
        return self.n

counter = Counter.remote()

@ray.remote
def do_work(counter, iters: int):
    for _ in range(iters):
        # ...some toy work...
        await_val = counter.add.remote(1)

# Launch several workers that all update the same counter
futures = [do_work.remote(counter, 1000) for _ in range(8)]
ray.get(futures)

total = ray.get(counter.value.remote())
print("Total count:", total)
```

Try:

- Changing the number of workers and iterations.
- Printing `counter.value()` while work is still running (`ray.wait` + periodic reads) to see concurrent updates.

### 3. Dask: parallelize a simple ETL pipeline

Simulate a small ETL job with `dask.dataframe`:

```python
import dask.dataframe as dd
import pandas as pd

# Create some fake CSV data
pdf = pd.DataFrame(
    {
        "user_id": range(1_000_000),
        "country": ["US", "CA", "FR", "DE"] * 250_000,
        "value": range(1_000_000),
    }
)
pdf.to_csv("events-*.csv", index=False)

# Load with Dask
df = dd.read_csv("events-*.csv", blocksize="16MiB")

filtered = df[df["country"] == "US"]
summary = (
    filtered.assign(log_value=dd.log(filtered["value"] + 1))
    .groupby("country")["log_value"]
    .mean()
)

print(summary.compute())
```

Open the Dask dashboard and pay attention to:

- The task stream: how many tasks, how long they run.
- Worker CPU and memory usage.

Experiment with:

- Different `blocksize` values.
- Changing the number of threads/processes via `dask.config.set` or the `Client` constructor.

### 4. Dask + Ray: data prep then training

Combine both frameworks in a minimal, two-step workflow:

1. Use Dask to preprocess and write a handful of shard files (small parquet or CSV).
2. Use Ray to launch one training-like task per shard that just “pretends” to train:

```python
# Step 1: Dask preprocessing (toy)
# - Read CSVs
# - Filter and add a feature
# - Write a few output shards

# Step 2: Ray "training"
import glob
import time

import ray

ray.init()

@ray.remote
def fake_train(shard_path: str):
    t0 = time.perf_counter()
    # ...read shard with pandas, sleep a bit, pretend to train...
    time.sleep(0.5)
    t1 = time.perf_counter()
    return {"shard": shard_path, "duration": t1 - t0}

shards = glob.glob("clean_shards/*.parquet")  # or *.csv
futures = [fake_train.remote(p) for p in shards]
results = ray.get(futures)
print(results)
```

Use this to reason about:

- Where you want boundaries between data prep and training.
- How you might pass real metrics back and aggregate them.

### 5. Reflect

After running at least two of these exercises, jot down:

- How your mental model of “task vs actor vs collection” changed.
- Which parts of the dashboards were most useful.
- One idea for how you could apply Ray or Dask to a real project you work on.

We’ll build on these patterns in later chapters when we combine multi-GPU training, multi-node clusters, and more complex orchestration.
