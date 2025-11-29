# Profiling and Benchmarking: Measuring Performance and Bottlenecks

This chapter is about turning “it feels slow” into concrete numbers. You’ll learn how to time code, profile CPU and GPU workloads, and interpret the results well enough to find real bottlenecks instead of guessing. The goal is to build a small toolkit of profiling habits you can reuse across the rest of the book.

Learning objectives:
- Distinguish between benchmarking, profiling, and tracing.
- Use simple timing tools (timers, `%timeit`, basic scripts) correctly.
- Run CPU and GPU profilers for common Python/AI workloads.
- Read profiler output to identify hot spots and bottlenecks.
- Design fair, repeatable benchmarks for model training and inference.
- Recognize common pitfalls (warmup, noise, I/O bottlenecks) and avoid misleading measurements.

Sections:
- Why profiling and benchmarking matter
- Timing basics: timers, `%timeit`, and microbenchmarks
- CPU profiling tools for Python workloads
- GPU and accelerator profiling basics
- Interpreting profiler output and spotting bottlenecks
- Designing fair, repeatable benchmarks
- Exercises: profiling a small model end-to-end

## Why profiling and benchmarking matter

When a program is slow, the natural instinct is to start optimizing whatever looks expensive—rewrite a loop, add a cache, move something to the GPU. Without measurements, this is guesswork. You might spend hours “optimizing” code that accounts for 2% of the total runtime while the real bottleneck sits elsewhere, untouched.

Profiling and benchmarking give you a map. **Profiling** tells you *where* time is actually spent: which functions, which lines, which kernels. **Benchmarking** tells you *how fast* a given configuration is under specific conditions. **Tracing** fills in the timeline between them: what ran *when*, and how different pieces overlapped (or failed to).

Together, these tools turn performance work from an art project into an engineering activity with feedback loops.

In AI workloads, this matters even more because the stakes are higher: training runs can take hours or days, and hardware is expensive. A small improvement in the right hotspot can save thousands of GPU-hours; a misguided optimization in the wrong place can waste the same amount of time without any visible effect.

Good profiling habits also prevent premature optimization. If a quick profile shows that 90% of your time is in data loading, you know not to spend a week hand-writing CUDA kernels. If a benchmark shows that switching hardware only saves 3% wall-clock time, you know that your bottleneck is somewhere else—maybe in I/O, maybe in Python overhead.

One practical way to think about this is as a feedback loop you run whenever performance matters:

> timing → CPU profiling → (optional) GPU/accelerator profiling and tracing → interpret results → design and run a fair benchmark → decide what to change → re-measure.

Each section in this chapter fills in one part of that loop and gives you a checklist you can reuse on new projects.

## Timing basics: timers, `%timeit`, and microbenchmarks

Before you reach for a full profiler, you should be comfortable with quick timing tools. They help answer questions like “is this change obviously faster or slower?” without a lot of ceremony.

The most portable option is a simple timer in Python:

```python
import time

t0 = time.perf_counter()
# --- code you want to measure ---
result = sum(i * i for i in range(10_000_000))
# --- end measured code ---
t1 = time.perf_counter()

print("Elapsed:", t1 - t0, "seconds")
```

`time.perf_counter()` is preferred over `time.time()` for short measurements because it has higher resolution and is intended for timing.

In interactive work (Jupyter, IPython), the `%timeit` magic gives you more robust microbenchmarks:

```python
# Single expression
%timeit [i * i for i in range(10_000)]

# A small code block
%%timeit
s = 0
for i in range(10_000):
    s += i * i
```

`%timeit` runs the code multiple times, throws away warmups, and reports average and standard deviation, which reduces noise. It’s ideal for comparing two small alternatives (e.g., list comprehension vs NumPy operation).

A few simple rules keep microbenchmarks honest:

- Benchmark **just the code** you care about, not imports or unrelated setup.
- Keep inputs realistic: timing a function on a size-10 array might hide problems that appear at size 10 million.
- For GPU code, always synchronize before stopping the timer, or use tools that do it for you.
- Repeat runs and look at variation; a noisy benchmark is usually telling you that something else (OS, I/O, background work) is interfering.

For example, with PyTorch on GPU:

```python
import time, torch

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(10_000_000, device=device)

# Warmup (JIT, kernels, caches)
for _ in range(5):
    _ = x * x
if device == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
y = x * x
if device == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()

print("GPU elapsed:", t1 - t0, "seconds")
```

These basic timing techniques won’t tell you *why* something is slow, but they are the first step in any performance investigation: they confirm that there is a problem, and they give you a baseline to compare against once you start profiling and optimizing. They also help you catch obvious issues like forgetting to warm up, accidentally including I/O in a compute benchmark, or measuring asynchronous GPU work without synchronization.

The next sections layer on profilers that explain *where* that time goes on both the CPU and GPU, so that later, when you build full benchmarks, you already know which parts of the workload deserve most of your attention.

## CPU profiling tools for Python workloads

Timers tell you *how long* a block of code takes. CPU profilers tell you *where that time goes*—which functions and lines are actually hot. The standard starting point in Python is `cProfile`, which is built into the standard library.

You can run `cProfile` from the command line:

```bash
python -m cProfile -o profile.out your_script.py
```

Then inspect it with `pstats`:

```python
import pstats

stats = pstats.Stats("profile.out")
stats.sort_stats("cumulative").print_stats(20)
```

This shows you which functions consumed the most cumulative time. For quick, inline profiling inside a script, you can also use the context manager form:

```python
import cProfile, pstats, io

def train_one_epoch():
    # ... your training or CPU-heavy code ...
    pass

pr = cProfile.Profile()
pr.enable()
train_one_epoch()
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(20)
print(s.getvalue())
```

For more visual exploration, tools like **snakeviz** or **tuna** can render `cProfile` output as interactive charts:

```bash
pip install snakeviz
python -m cProfile -o profile.out your_script.py
snakeviz profile.out
```

Basic tips when CPU profiling:

- Profile a **realistic workload** (enough data/steps to be representative, but not so large that profiling takes hours).
- Start broad: profile the whole script once, then zoom into hotspots with more targeted profiles.
- Watch out for noise from logging, debug printing, or one-off initialization—profile the steady-state part of your workload when possible.
- Avoid over-instrumenting: every extra layer of tracing adds overhead and can distort timings; start with `cProfile` and only reach for lower-level tools if needed.

In later sections we’ll complement this with GPU and accelerator profilers and timeline-style traces, but for many Python-heavy pipelines (data loading, preprocessing, orchestration), `cProfile` plus a visualizer is enough to find the biggest wins. Once you have a sense of where CPU time goes, you can carry that context into the GPU/accelerator view and, eventually, into the higher-level “interpret and benchmark” loop.

## GPU and accelerator profiling basics

CPU profilers tell you what Python and native code are doing on the host. For GPU and other accelerators, you also care about *device-side* work: which kernels run, how busy the device is, and where time is lost to transfers and synchronization.

Most deep learning frameworks expose at least two layers of tooling:

- A **high-level profiler** inside the framework (PyTorch, TensorFlow, JAX).
- **Vendor tools** (NVIDIA Nsight Systems/Compute, ROCm tools, TPU profilers) for detailed kernel-level analysis and timeline-style tracing.

A simple starting point with PyTorch looks like this:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.nn.Sequential(
    torch.nn.Linear(2048, 2048),
    torch.nn.ReLU(),
    torch.nn.Linear(2048, 2048),
).to(device)

x = torch.randn(512, 2048, device=device)

# Warmup to avoid initialization skew
for _ in range(5):
    y = model(x)
    loss = y.sum()
    loss.backward()
    model.zero_grad(set_to_none=True)
if device == "cuda":
    torch.cuda.synchronize()

with torch.autograd.profiler.profile(use_cuda=(device == "cuda")) as prof:
    for _ in range(10):
        y = model(x)
        loss = y.sum()
        loss.backward()
        model.zero_grad(set_to_none=True)

print(prof.key_averages().table(
    sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total",
    row_limit=10,
))
```

In newer PyTorch, you can also use `torch.profiler` (which integrates with TensorBoard and can produce traces):

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Linear(4096, 4096).to(device)
x = torch.randn(1024, 4096, device=device)

def step():
    y = model(x)
    y.sum().backward()
    model.zero_grad(set_to_none=True)

# Warmup
for _ in range(10):
    step()
if device == "cuda":
    torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logdir"),
) as prof:
    for _ in range(20):
        step()
        prof.step()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
```

General tips for GPU/accelerator profiling:

- **Warm up** first: run a few iterations before starting the profiler to avoid one-time setup dominating results.
- **Keep runs short**: profiling adds overhead; capture dozens or hundreds of iterations, not entire multi-hour trainings.
- **Watch transfers**: time spent in `memcpy`/host-device transfer is often a sign of inefficient input pipelines or frequent `.to(device)` calls.
- Use timeline views (from `torch.profiler` traces or Nsight Systems) when you need to see how CPU, GPU, and data loading overlap in time—not just how long individual ops take.

When you need more detail (per-kernel breakdowns, occupancy, memory bandwidth), you can run vendor tools around your script (e.g., `nsys profile python train.py` for NVIDIA). This book will introduce those tools where needed, but for many day-to-day questions, the framework’s built-in GPU profiler is enough to find obvious bottlenecks. In the next section we’ll use the tables and traces produced by both CPU and GPU profilers and show how to read them in a unified way to decide which hotspots to tackle first.

## Interpreting profiler output and spotting bottlenecks

Running a profiler is the easy part; the real value comes from deciding *what to do* with the numbers it prints. The goal is not to stare at every line, but to quickly identify a few candidates that are worth optimizing, whether they come from CPU (`cProfile`) or GPU (`torch.profiler`) runs.

For CPU profiles (e.g., `cProfile`, `pstats`), start with a table sorted by cumulative time:

```python
stats.sort_stats("cumulative").print_stats(20)
```

Look for:

- Functions that account for a large fraction of total time (e.g., 30–80%).
- Hot functions you *own* (in your modules), not low-level library calls you can’t easily change.
- Surprises: places you didn’t expect to be expensive at all.

For GPU profiles (e.g., `torch.profiler`), sort by CUDA time:

```python
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=10,
))
```

Here you’re asking:

- Which kernels or ops dominate device time?
- How much time is spent in `memcpy` / host-device transfers versus actual compute?
- Are there many tiny kernels (launch overhead) instead of fewer, larger ones?

If you have a trace view (from `torch.profiler` or a vendor tool), also glance at the timeline:

- Is the GPU idle while the CPU loads data?
- Are there large gaps between kernels that indicate synchronization or Python overhead?
- Do different streams (compute, transfers) overlap as you expect?

To make these tables more concrete, imagine a tiny `cProfile` summary like:

```text
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    50    0.010    0.000    1.200    0.024 myproj/data.py:42(load_batch)
  1000    0.300    0.000    0.800    0.001 myproj/model.py:10(forward)
    50    0.005    0.000    0.050    0.001 myproj/train.py:88(log_metrics)
```

In this view:

- `cumtime` is the column to sort by when you want to see “where total time goes” (here, `load_batch` dominates).
- `ncalls` and `percall` help you see if a function is slow because it’s called many times or because each call is individually expensive.
- Low-percentage entries like `log_metrics` (small `cumtime`) are usually not worth optimizing until the top lines are under control.

The same logic applies to GPU `key_averages().table(...)` output: you focus on rows with large cumulative CUDA time, especially those corresponding to operations you call frequently or control directly.

Once you have a hotspot, work backwards:

- If the profiler says `my_project.data.load_batch` is hot, can you batch I/O, cache results, prefetch in a background worker, or move some work off the critical path?
- If a single matmul or convolution dominates, can you reduce its frequency (e.g., fewer layers/steps), change its shape (batch size, sequence length), or use a more efficient implementation (mixed precision, fused kernels)?
- If traces show the GPU is idle while the CPU prepares data, can you overlap work with asynchronous data loading or larger batches?

A few heuristics help you avoid chasing noise:

- Focus first on the top 1–3 entries that together account for most of the time.
- Ignore micro-optimizing code that’s <5% of total runtime unless everything else is already tuned.
- Re-run the same profile after each change to confirm that the hotspot actually shrank and that you didn’t just move the bottleneck somewhere else.
- Compare profiles at different batch sizes or model configurations to see how hotspots scale; some issues only appear at “real” problem sizes.

As you work through later chapters, you’ll repeat this loop often: profile, identify a clear bottleneck (on CPU, GPU, or both), try a targeted change, and re-measure. Developing that habit matters more than memorizing any specific profiler option, and it plugs directly into the benchmarking patterns you’ll see next.

## Designing fair, repeatable benchmarks

Once you know where time is going, you often want to compare alternatives: two implementations, two library versions, or two hardware setups. To make those comparisons meaningful, you need benchmarks that are *fair* and *repeatable*.

A fair benchmark keeps as many variables fixed as possible:

- Same input sizes and shapes.
- Same number of warmup iterations.
- Same number of timed iterations.
- Same hardware configuration (device type, number of threads, environment variables).
- Same work definition (e.g., “one training step” vs “one epoch”) across all variants.

A simple pattern for a Python+NumPy benchmark:

```python
import time
import numpy as np

def bench(fn, *, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)  # or np.median(times)

N = 1_000_000
x = np.random.randn(N).astype(np.float32)

def baseline():
    return np.square(x)

def alternative():
    return x * x

print("baseline:", bench(baseline))
print("alternative:", bench(alternative))
```

Key points:

- Wrap each candidate in a function that does *only* the work you want to measure.
- Run each candidate multiple times and use the best or median time to reduce noise.
- Avoid including one-off setup (data loading, model construction) in the timed region unless that’s what you care about.
- Make sure you’re not accidentally benchmarking different work (e.g., different batch sizes) when you compare numbers.

For training loops or GPU code, you can follow the same pattern with a tiny harness:

```python
import time, torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.nn.Linear(4096, 4096).to(device)
x = torch.randn(1024, 4096, device=device)

def step():
    y = model(x)
    loss = y.square().mean()
    loss.backward()
    model.zero_grad(set_to_none=True)

# Warmup
for _ in range(10):
    step()
if device == "cuda":
    torch.cuda.synchronize()

def bench(fn, steps=50, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(steps):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) / steps)
    return min(times)

print("seconds per step:", bench(step))
```

To make benchmarks *repeatable*:

- Fix random seeds where randomness matters (`torch.manual_seed`, `np.random.seed`, etc.).
- Record key environment details alongside results: library versions, GPU model, batch size, sequence length, number of data loader workers, etc.
- Automate them: put benchmark scripts under `benchmarks/` or `scripts/` and run them with a single command.
- Re-run occasionally to catch regressions when dependencies or hardware change.

Over time you can treat these benchmarks like tests for performance: when you change code, you re-run them to check whether things got faster, slower, or stayed the same. Later chapters will reuse this pattern to compare CPUs vs GPUs, different parallelism strategies, and different model architectures under controlled conditions.

## Exercises: profiling a small model end-to-end

These exercises tie together timing, profiling, and basic benchmarking. Use whichever framework you’re comfortable with; examples below use PyTorch.

### 1. Baseline timing of a training step

1. Create a tiny model and dataset:

   ```python
   import time, torch

   device = "cuda" if torch.cuda.is_available() else "cpu"

   model = torch.nn.Sequential(
       torch.nn.Linear(784, 512),
       torch.nn.ReLU(),
       torch.nn.Linear(512, 10),
   ).to(device)

   x = torch.randn(256, 784, device=device)
   y = torch.randint(0, 10, (256,), device=device)

   opt = torch.optim.SGD(model.parameters(), lr=0.1)
   loss_fn = torch.nn.CrossEntropyLoss()
   ```

2. Time an average training step:

   ```python
   def step():
       opt.zero_grad(set_to_none=True)
       logits = model(x)
       loss = loss_fn(logits, y)
       loss.backward()
       opt.step()

   # Warmup
   for _ in range(10):
       step()
   if device == "cuda":
       torch.cuda.synchronize()

   iters = 50
   t0 = time.perf_counter()
   for _ in range(iters):
       step()
   if device == "cuda":
       torch.cuda.synchronize()
   t1 = time.perf_counter()

   print("Seconds per step:", (t1 - t0) / iters)
   ```

3. Optional: repeat the run a couple of times and note how stable the “seconds per step” number is.

Record this number; you’ll compare against it later.

### 2. CPU profile of the training loop

Wrap the same `step()` loop in `cProfile`:

```python
import cProfile, pstats, io

pr = cProfile.Profile()
pr.enable()
for _ in range(50):
    step()
if device == "cuda":
    torch.cuda.synchronize()
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(20)
print(s.getvalue())
```

Questions to answer:

- Which functions at the Python level take the most time?
- Are there any obvious surprises (e.g., data conversion, logging)?
- Does the profile look different if you increase `iters` or batch size?

### 3. GPU profile of the same workload (if you have a GPU)

If you’re on GPU, run a short `torch.profiler` session:

```python
import torch

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(20):
        step()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

If you configure `on_trace_ready` and a log directory, open the trace in TensorBoard and inspect the timeline.

Questions:

- Which ops or kernels dominate CUDA time?
- How much time is in `memcpy` / transfers vs pure compute?
- Does the timeline show the GPU waiting on the CPU or data loader?

### 4. Try one small change and re-measure

Pick a tiny, safe change, such as:

- Change batch size (e.g., 128 → 512).
- Swap optimizer (SGD ↔ Adam).
- Turn on `torch.backends.cudnn.benchmark = True` for convolutional models.
- Add workers to your data loader if you’re using one.

Then:

1. Re-run the timing loop and note the new seconds/step.
2. Re-run the profiler briefly and see if the hotspot list changed.
3. Decide whether the change is worth keeping based on both speed and code complexity.

The goal is not to find the perfect optimization, just to practice the loop:

> measure → inspect profiler output or trace → change one thing → measure again.

You’ll repeat this pattern throughout the rest of the book on larger and more complex workloads, but the basic habits you build here will carry over unchanged.
