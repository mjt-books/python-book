# CPU Optimizations and Multi-threading

This chapter focuses on getting more out of your CPUs: using vectorized operations, multi-threading, and asynchronous I/O to speed up Python and AI workloads. You’ll see how to lean on libraries that use SIMD and OpenMP under the hood, when to reach for `multiprocessing` or native threads, and how `asyncio` can help hide I/O latency.

Learning objectives:
- Understand how CPU vectorization (SIMD) accelerates numerical workloads.
- Recognize when libraries already use multi-threading (e.g., BLAS, OpenMP-backed code).
- Use basic Python tools for parallelism (threads, processes, `concurrent.futures`).
- Apply `asyncio` to overlap I/O-bound work with computation.
- Measure and reason about CPU utilization and scaling limits (Amdahl’s law, contention).

Sections:
- Why CPU optimization still matters in an accelerator world
- Vectorization basics: NumPy, broadcasting, and avoiding Python loops
- Multi-threading and multi-processing in Python (and their limits)
- Native libraries, BLAS, and OpenMP: letting C/C++ do the heavy lifting
- Async I/O with `asyncio` for data pipelines
- Measuring CPU utilization and scaling behavior
- Exercises: speeding up a CPU-bound data preprocessing pipeline

## Why CPU optimization still matters in an accelerator world

With GPUs, TPUs, and other accelerators everywhere, it’s easy to think of CPUs as just “the thing that launches kernels.” In practice, your CPUs are still doing a lot of work: reading data from disk, decoding and augmenting samples, coordinating processes, talking to the network, logging metrics, and driving the host-side parts of your framework. If that work is slow or poorly parallelized, your shiny accelerators sit idle.

Most real training and inference pipelines look like a *chain*:

> storage → CPU-heavy preprocessing → host/device transfers → GPU/TPU compute → postprocessing → logging

Optimizing only the GPU segment can hit a hard ceiling if the CPU stages can’t keep up. Typical symptoms include low GPU utilization, data loader bottlenecks, and models that speed up on paper (more FLOPs/sec) but don’t actually shorten end-to-end wall‑clock time.

CPUs also matter in scenarios where accelerators are scarce or unavailable: small teams without a GPU budget, edge deployments on CPU-only servers, or large-scale preprocessing jobs that prepare data for later training. In those settings, vectorization and multi-threading can easily yield 2–10× speedups without changing hardware.

Finally, understanding CPU behavior helps you reason about the whole system: how many data-loader workers to use, how to set thread counts for BLAS/OpenMP libraries, when to offload tasks to separate processes, and how to overlap I/O with compute. Later chapters will build on this, but the core message here is simple:

> Even in an accelerator-first world, CPUs set the pace for everything around your model. If you ignore them, you leave performance—and money—on the table.

## Vectorization basics: NumPy, broadcasting, and avoiding Python loops

The easiest way to speed up CPU-bound numerical code in Python is to *stop writing the inner loops in Python*. Vectorized libraries like NumPy execute those loops in optimized C with SIMD instructions and multi-threading, while Python handles only high-level orchestration.

Consider a simple example: computing the L2 norm of each row in a large matrix.

A pure-Python version:

```python
import math

def row_norms_python(x):
    # x: list of lists (rows)
    out = []
    for row in x:
        s = 0.0
        for v in row:
            s += v * v
        out.append(math.sqrt(s))
    return out
```

A vectorized NumPy version:

```python
import numpy as np

def row_norms_numpy(x):
    # x: NumPy array of shape (n_rows, n_cols)
    # Square, sum along columns, then sqrt
    return np.sqrt((x * x).sum(axis=1))
```

On large inputs, `row_norms_numpy` is typically orders of magnitude faster, because:

- The inner loops run in C instead of Python.
- NumPy can use SIMD instructions (e.g., AVX/AVX-512).
- NumPy can leverage multi-threaded BLAS/OpenMP where appropriate.

Broadcasting lets you apply operations across arrays of different but compatible shapes without explicit Python loops. For example, subtracting a per-feature mean from every row:

```python
x = np.random.randn(100_000, 256)   # batch of feature vectors
mean = x.mean(axis=0)               # shape: (256,)

# Broadcasting: mean is "stretched" along axis 0
x_centered = x - mean
```

No loops in Python, but effectively you’ve done 100,000 vector subtractions.

A few practical guidelines:

- **Think in arrays, not scalars**: whenever you see a `for` loop touching every element of a large list/array, ask if it can become a single NumPy expression.
- **Fuse operations**: `y = (x * x + 1.0).sum(axis=1)` is better than a series of intermediate Python-level loops and temporaries.
- **Watch memory**: vectorization can create large temporary arrays. Use in-place ops (`x *= 2`) or functions like `np.einsum`/`np.dot` when appropriate to keep things compact.

We’ll use this mental model throughout the chapter: first try to express the work as a few high-level array operations; only if that fails do we consider more explicit threading or lower-level optimizations.

## Multi-threading and multi-processing in Python (and their limits)

Vectorization should be your first move for numeric workloads, but sometimes you still need *concurrency*: running multiple pieces of work at the same time. In Python, this usually means choosing between **threads** and **processes**, with the Global Interpreter Lock (GIL) shaping what’s possible.

The rough mental model:

- **Threads** share memory inside one process; cheap to create, good for I/O-bound tasks, but CPU-bound pure-Python code doesn’t scale well because of the GIL.
- **Processes** have separate memory; heavier to spawn, but each has its own interpreter, so CPU-bound code can run in parallel across cores.

A minimal example with `concurrent.futures.ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor
import requests

urls = ["https://example.com"] * 20

def fetch(url):
    resp = requests.get(url, timeout=5)
    return len(resp.content)

with ThreadPoolExecutor(max_workers=8) as pool:
    sizes = list(pool.map(fetch, urls))
print("Total bytes:", sum(sizes))
```

This speeds up I/O-bound workloads (network, disk) by overlapping waits. The GIL is not a big issue here because most time is spent in C extensions or the OS, which can release the GIL.

For CPU-bound Python functions, use `ProcessPoolExecutor` instead:

```python
from concurrent.futures import ProcessPoolExecutor
import math

def heavy_compute(n: int) -> float:
    s = 0.0
    for i in range(n):
        s += math.sqrt(i)
    return s

N = 5_000_00  # adjust as needed
tasks = [N] * 8

with ProcessPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(heavy_compute, tasks))

print("Results:", results)
```

Here each worker runs in its own process, so work can truly occupy multiple cores.

A few practical points and limits:

- **Overhead**: Spawning many processes and shipping large arrays between them can cost more than you gain. Aim for *fewer, chunkier* tasks rather than thousands of tiny ones.
- **Shared state**: Threads share memory but require careful synchronization (locks, queues). Processes don’t share memory by default, so you pass data via pickling or shared memory primitives.
- **GIL-friendly code**: Many numeric libraries (NumPy, PyTorch, etc.) release the GIL while doing heavy work in C. In those cases, threads can still help, because the true bottleneck isn’t the Python interpreter.

In this chapter we’ll mostly use threads for I/O-bound parts of the pipeline (data loading, prefetching, logging) and processes for CPU-heavy Python code that can’t easily be vectorized. Later sections will connect this to native libraries (BLAS, OpenMP) that give you multi-threading “for free” under the hood.

## Native libraries, BLAS, and OpenMP: letting C/C++ do the heavy lifting

So far we’ve focused on what *your* Python code can do. A lot of the real speed, however, comes from native libraries underneath: BLAS implementations (OpenBLAS, MKL, BLIS), FFT libraries, and C/C++ code parallelized with OpenMP or similar APIs. NumPy, SciPy, PyTorch, TensorFlow, and many other libraries already use these for you.

For example, a single call like:

```python
import numpy as np

A = np.random.randn(10_000, 4_096)
B = np.random.randn(4_096, 1_024)

C = A @ B  # matrix multiply
```

typically invokes highly optimized BLAS routines in C/Fortran that:

- Use SIMD instructions across vector lanes.
- Spawn multiple threads internally to use all cores.
- Carefully manage cache and memory access patterns.

From Python’s point of view, this is *one* operation, but under the hood it can saturate the CPU.

You can often control how many threads these libraries use via environment variables:

```bash
# Common knobs (exact behavior depends on your BLAS build):
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

It’s worth experimenting: too few threads underutilizes the CPU; too many can cause contention, context switching, and worse performance—especially if you also use Python threads or multiple processes.

A few practical guidelines:

- **Prefer library calls over manual loops**: if a problem can be expressed as matrix multiplies, convolutions, or reductions, let BLAS or other native kernels handle it.
- **Avoid “double parallelism”**: don’t spawn N Python workers each calling an N-threaded BLAS routine; you’ll end up with N×N threads fighting for the same cores. Either reduce BLAS threads per process or reduce the number of processes.
- **Know your stack**: check which BLAS your NumPy/PyTorch build uses (`np.__config__.show()`, PyTorch build logs) so you know which env vars actually matter.

The big idea is that you rarely need to write OpenMP or SIMD intrinsics yourself. Instead, you:

1. Express work in terms of high-level operations that map to fast native kernels.
2. Set a few environment variables or config options so those kernels use your CPU effectively.
3. Measure with the profiling techniques from chapter 3 to confirm that CPU cores are busy and not oversubscribed.

## Async I/O with `asyncio` for data pipelines

Not all bottlenecks are compute-bound. In many real systems, CPUs spend a lot of time *waiting*: for disks, networks, message queues, or remote APIs. When that’s the case, throwing more threads or processes at the problem works, but it can get heavy. `asyncio` gives you another option: a single event loop that juggles many I/O-bound tasks cooperatively.

The core idea is simple:

- Mark functions that perform I/O as `async def`.
- Use `await` when you hit an operation that can yield (reading from disk/socket, sleeping, etc.).
- Use `asyncio.gather` (or similar) to run many such tasks concurrently.

A minimal sketch for an asynchronous data fetch + preprocess pipeline:

```python
import asyncio
import aiohttp  # third-party: pip install aiohttp

async def fetch_json(session, url):
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.json()

async def preprocess(record):
    # Pretend this is CPU-light but frequent work
    return {k: str(v).lower() for k, v in record.items()}

async def load_batch(urls):
    async with aiohttp.ClientSession() as session:
        # Fire off all fetches concurrently
        fetched = await asyncio.gather(
            *(fetch_json(session, u) for u in urls)
        )
        # Preprocess concurrently too (still one thread, but overlaps I/O)
        processed = await asyncio.gather(
            *(preprocess(r) for r in fetched)
        )
        return processed

async def main():
    urls = ["https://httpbin.org/json"] * 32
    batch = await load_batch(urls)
    print("Loaded", len(batch), "records")

asyncio.run(main())
```

In an AI context, `asyncio` is most useful when:

- You have to talk to remote storage or services (object stores, feature stores, HTTP APIs).
- Each individual I/O operation is small, but you have many of them per step.
- CPU work per I/O request is modest (heavy CPU work is usually better off in threads/processes or native code).

Practical tips:

- Keep async functions mostly I/O-bound; if they become CPU-heavy, move the heavy part to a thread/process pool (`loop.run_in_executor` or `asyncio.to_thread`).
- Don’t mix `asyncio` and thread/process pools casually without measuring—complexity adds up quickly.
- Treat `asyncio` as another way to keep CPUs busy *while they wait*, not as a replacement for vectorization or proper multi-threaded native libraries.

In later chapters, when we build more complex input pipelines, you’ll see how to combine `asyncio`, threads, and native code so that data keeps flowing fast enough to stay ahead of your accelerators.

## Measuring CPU utilization and scaling behavior

Optimizing CPU code without looking at CPU usage is like tuning a car engine with your eyes closed. You need at least a rough sense of how busy your cores are and how performance scales as you add threads, processes, or data-loader workers.

On a Linux machine, start with simple system tools:

```bash
top              # or: htop
```

Watch:

- **Overall CPU%**: are your CPUs mostly idle or near 100%?
- **Per-core usage** (htop): is work spread across cores or stuck on one?
- **Per-process CPU%**: is your Python process actually consuming CPU when you expect it to?

From Python, `psutil` can give you quick programmatic checks:

```python
import time
import psutil

proc = psutil.Process()
for _ in range(5):
    cpu = proc.cpu_percent(interval=1.0)
    print("Process CPU%:", cpu)
```

Run this around a workload to confirm that CPU usage rises when your code is active.

To reason about *scaling*, measure how runtime changes as you vary a single knob—for example, number of worker processes:

```python
import time
from concurrent.futures import ProcessPoolExecutor

def work(n: int) -> int:
    s = 0
    for i in range(n):
        s += i * i
    return s

def run_with_workers(workers: int, tasks: int = 8, n: int = 2_000_000) -> float:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, [n] * tasks))
    t1 = time.perf_counter()
    return t1 - t0

for w in (1, 2, 4, 8):
    elapsed = run_with_workers(w)
    print(f"{w} workers: {elapsed:.2f} s")
```

Questions to ask yourself:

- Do you get close to 2× speedup going from 1 → 2 workers? From 2 → 4?
- Where does scaling flatten or get worse? That’s often where you hit contention (CPU, memory bandwidth, disk, GIL, etc.).

This connects directly to Amdahl’s law: if 30% of your workload is inherently serial (e.g., single-threaded data loading, global locks, coordination), no amount of extra cores will give you more than ~3.3× speedup. Rough numbers from these scaling experiments help you decide whether to:

- Focus on parallelizing more of the pipeline, or
- Accept the limit and move the heavy part to a GPU/accelerator instead.

In the final section of this chapter, you’ll apply these ideas to a small data preprocessing pipeline and measure how far simple vectorization and concurrency can take you on a single machine.

## Exercises: speeding up a CPU-bound data preprocessing pipeline

These exercises give you hands-on practice with vectorization and CPU concurrency. Keep them small; the goal is to *feel* the speedups, not build a full pipeline.

### 1. Replace a Python loop with NumPy

Start with a pure-Python preprocessing step:

```python
import math
import random
import time

def preprocess_python(xs):
    out = []
    for x in xs:
        y = math.tanh(x) * 2.0 + 1.0
        out.append(y)
    return out

N = 5_000_000
xs = [random.random() for _ in range(N)]

t0 = time.perf_counter()
ys = preprocess_python(xs)
t1 = time.perf_counter()
print("Python loop:", t1 - t0, "s")
```

Then rewrite it using NumPy:

```python
import numpy as np
import time

xs_np = np.array(xs, dtype=np.float32)

t0 = time.perf_counter()
ys_np = np.tanh(xs_np) * 2.0 + 1.0
t1 = time.perf_counter()
print("NumPy vectorized:", t1 - t0, "s")
```

Compare times and confirm that results are close (`np.allclose`).

### 2. Parallelize a CPU-heavy function with processes

Take a CPU-bound function:

```python
import math

def heavy_compute(n: int) -> float:
    s = 0.0
    for i in range(n):
        s += math.sqrt(i)
    return s
```

Time it sequentially vs with `ProcessPoolExecutor`:

```python
import time
from concurrent.futures import ProcessPoolExecutor

def run_seq(tasks=4, n=2_000_000):
    t0 = time.perf_counter()
    results = [heavy_compute(n) for _ in range(tasks)]
    t1 = time.perf_counter()
    print("Sequential:", t1 - t0, "s")
    return results

def run_par(workers=4, tasks=4, n=2_000_000):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(heavy_compute, [n] * tasks))
    t1 = time.perf_counter()
    print(f"{workers} workers:", t1 - t0, "s")
    return results

run_seq()
run_par(workers=2)
run_par(workers=4)
```

Observe how speedup changes with the number of workers.

### 3. Observe CPU utilization while scaling workers

Re-run exercise 2, but this time watch CPU usage with `htop` or `top` as you change `workers` (1, 2, 4, 8). Answer for yourself:

- At what point do extra workers stop helping?
- Do all cores get used, or are some idle?

### 4. Optional: add async I/O around a small CPU core

Mock a tiny pipeline where you:

- `async`-fetch or `async`-sleep to simulate I/O for N items.
- Use `asyncio.to_thread` (or a thread pool) to run a small CPU-bound function on each.
- Measure how total time changes vs a purely sequential version.

Even a toy example is enough to see how overlapping I/O and CPU work can keep your machine busier without changing hardware.
