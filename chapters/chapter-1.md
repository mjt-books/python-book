# Foundations: AI, Python, and the Hardware Landscape

This chapter establishes the foundational concepts needed throughout the book: the current AI landscape, why hardware choices matter, and how Python acts as the bridge between models and devices.

Learning objectives:
- Explain the roles of CPUs, GPUs, TPUs, and edge accelerators in modern AI workflows.
- Describe how Python interfaces with hardware (libraries, runtimes, drivers).
- Identify common trade-offs when selecting hardware for training vs inference.
- Prepare the environment and tools used throughout the book.

Sections:
- Overview: why hardware matters for AI
- Python ecosystem: key libraries and runtimes
- Hardware primer: CPUs, GPUs, TPUs, and edge devices
- Performance trade-offs and cost considerations
- Quick setup: verifying device availability and drivers
- Exercises: measuring simple kernel performance and comparing runtimes

## Overview: why hardware matters for AI

If you train a small model on your laptop and a giant model on a datacenter cluster, you are still doing the same conceptual thing: multiplying tensors and moving bytes around. The main difference is *how fast* you can do that and *how expensive* it is. Hardware is where those differences come from. The same Python code can feel instant on one machine and painfully slow on another.

Modern AI workloads are dominated by a few simple operations (matrix multiplies, convolutions, elementwise kernels) applied at huge scale. CPUs, GPUs, TPUs, and other accelerators all implement these operations, but they have different strengths: CPUs are flexible and great for control-heavy code, GPUs excel at massively parallel math, and TPUs and other accelerators trade flexibility for even higher throughput on specific workloads.

Ignoring hardware details works for toy problems, but it quickly breaks down in practice. Training times stretch from minutes to days, memory fills up mid-experiment, or inference latency blows past your product’s SLA. When that happens, “just write Python” is not enough—you need to understand what the hardware is doing underneath and how to steer it without rewriting everything in C++.

The good news is that you rarely need to write low-level kernels yourself. Instead, you’ll learn how to *ask the hardware better questions*: choose batch sizes that keep devices busy, structure your data loading so CPUs and GPUs overlap productively, and select the right mix of precision, memory, and parallelism. Python becomes the orchestrator, gluing together high-performance libraries and multiple devices.

Throughout this book we’ll keep returning to the same practical question: *given this model and this hardware, how do we get the most useful work done per unit of time, energy, and money?* This chapter gives you the mental model to answer that question, so later chapters on GPUs, TPUs, clusters, and edge devices will feel like variations on a familiar theme rather than entirely new worlds.

## Python ecosystem: key libraries and runtimes

Python’s superpower in AI is not raw speed; it’s delegation. Most of the heavy lifting happens in optimized C, C++, CUDA, or specialized runtimes, while Python coordinates *what* to run and *where*. Understanding the major pieces of this ecosystem will help you know which tool to reach for when you want to light up more hardware.

At the numerical core you’ll find **NumPy**, **SciPy**, and **pandas**. These libraries are CPU-focused and great for prototyping, data wrangling, and smaller models. Many GPU-accelerated libraries copy the NumPy API (for example, CuPy or PyTorch tensors), so getting comfortable with NumPy-style array code pays off everywhere else.

For deep learning, **PyTorch**, **TensorFlow**, and **JAX** dominate. They provide high-level tensor operations, automatic differentiation, and integrations with GPUs and other accelerators. Under the hood they call vendor libraries like cuBLAS and cuDNN, and they include their own runtimes to schedule work across devices. When you write `tensor.to("cuda")` or `pmap` a function in JAX, you’re really instructing these runtimes how to use your hardware.

On the deployment side, formats and runtimes like **ONNX**, **TensorRT**, and **TorchScript** make it possible to export models and run them efficiently on servers, edge devices, or specialized accelerators. These tools focus on inference: optimizing graphs, fusing operations, and targeting different backends without forcing you to rewrite your model from scratch.

Finally, distributed compute frameworks such as **Ray**, **Dask**, and libraries built on top of them help you scale beyond a single machine. They coordinate multiple processes, nodes, and devices so that large training runs, hyperparameter searches, or data preprocessing pipelines can make full use of available resources.

In the rest of this book, we’ll treat Python as the “control plane” that talks to these libraries and runtimes. You don’t need to memorize every API, but you should recognize the roles they play: numerical building blocks, training frameworks, inference runtimes, and distributed schedulers. We’ll focus on how to combine them to turn a single-machine script into a scalable, hardware-aware system.

## Hardware primer: CPUs, GPUs, TPUs, and edge devices

At a high level, all of these devices do the same job: perform lots of simple math on arrays of numbers. The differences are in how much parallel work they can handle, how flexible they are, how much memory they have, and how easy they are to program. Having a rough mental model of each type will make hardware choices and debugging far less mysterious.

**CPUs** (central processing units) are the generalists. They handle operating systems, Python interpreters, data loading, logging, and all the glue code around your model. Modern CPUs have multiple cores and vector units (SIMD), so they can still do a fair amount of numerical work, especially when code is well-vectorized. They shine when control flow is complex, batch sizes are small, or you’re juggling many different tasks.

**GPUs** (graphics processing units) are the specialists for throughput. They contain thousands of smaller cores designed to run the same operation on many data elements in parallel. That makes them ideal for dense linear algebra—matrix multiplications and convolutions—which dominate modern deep learning. The trade-offs: they want large, regular workloads to stay busy, they’re usually attached over a bus (like PCIe), and their on-device memory is fast but limited.

**TPUs and other AI accelerators** push specialization even further. They are built around tensor/matrix units and on-chip memory hierarchies tuned for deep learning workloads. You typically access them through higher-level frameworks (e.g., JAX/XLA, TensorFlow) rather than low-level kernels. In return, you get very high performance and good scaling—if your model fits their execution model and memory constraints.

**Edge devices** (phones, embedded boards, small form factor GPUs, NPUs in laptops, etc.) live under stricter power, thermal, and memory limits. Here, the problem is less “how do I finish training in 8 hours instead of 2 days?” and more “how do I serve predictions reliably at a few watts?” Tooling like ONNX Runtime, TensorRT, Core ML, or vendor-specific SDKs help squeeze models into these constraints via quantization, pruning, and specialized kernels.

In practice, you’ll often combine several of these: CPUs feeding data to GPUs, GPUs or TPUs doing the main training, and edge devices running optimized versions of the model in production. The rest of this book will show you how to design code and workflows that can move relatively smoothly across this spectrum instead of being locked into a single device type.

## Performance trade-offs and cost considerations

Scaling AI is always a set of trade-offs: you can make things faster, cheaper, or simpler to manage—but rarely all three at once. Choosing hardware is really about choosing where you’re willing to pay: in wall-clock time, in dollars, in energy, or in engineering effort.

On the performance side, more powerful hardware (bigger GPUs, more nodes, specialized accelerators) usually shrinks training time and inference latency, up to a point. But pushing for maximum throughput can increase queue times, failure rates, and debugging complexity. Sometimes it is faster *in calendar time* to run slightly slower jobs on hardware that is easier to get and easier to reason about.

Memory is another axis. Larger devices let you use bigger models and batch sizes, which improves utilization and often stability. But they also cost more per hour, and wasting that memory with inefficient data pipelines or poorly chosen batch sizes can erase any theoretical advantage. A well-tuned mid-range GPU can beat a top-tier GPU that is starved for data or stuck at low utilization.

Cost and energy usage matter as soon as you leave the toy stage. A model that trains in 4 hours on one GPU and in 1 hour on four GPUs doesn’t automatically justify 4× the cost. For production inference, shaving a few milliseconds of latency may or may not be worth doubling your hardware bill. You want a rough feel for when adding more hardware is a bargain and when it is just expensive decoration.

Finally, there is the cost of *your* time and your team’s complexity budget. Techniques like quantization, model parallelism, or aggressive pipeline optimizations can deliver big wins, but they also introduce more moving parts, more code paths, and more failure modes. Often, the best first step is to use simple, well-understood patterns—good batching, decent input pipelines, basic profiling—and only reach for advanced tricks when profiling says you must.

Throughout this book, we’ll keep these trade-offs in view. When we introduce a new optimization or hardware setup, we’ll ask: what performance does it buy, what does it cost in money and energy, and how much extra complexity does it add? The goal is not to make everything as fast as possible, but to make it fast *enough* for your constraints, with a clear understanding of what you’re paying for.

## Quick setup: verifying device availability and drivers

Before we dive into serious optimization, it’s worth confirming what hardware you actually have and whether the right drivers and libraries can see it. The goal here is not full troubleshooting, just a quick “sanity check” so you don’t spend hours tuning code that can’t reach your GPU or accelerator.

On a Linux machine, you can start at the OS level:

```bash
# List CPUs
lscpu | head

# List PCI devices and look for GPUs/accelerators
lspci | grep -i -E "nvidia|amd|graphics|accelerator" || true
```

If you expect NVIDIA GPUs, the `nvidia-smi` tool is your friend:

```bash
nvidia-smi
```

This should show one row per GPU, including memory, utilization, and driver version. If it fails or returns “command not found”, either the NVIDIA driver or CUDA toolkit is not installed correctly (later chapters will cover recommended setups).

From Python, you can perform similar checks using common libraries:

```python
# Basic CPU and Python environment info
import platform, sys
print(platform.platform())
print("Python:", sys.version)

# PyTorch GPU check
import torch
print("PyTorch CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))

# TensorFlow GPU check
import tensorflow as tf
print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))
```

For JAX/TPU or other accelerators, you’ll typically be running in a managed environment (e.g., cloud TPUs). There, the platform often exposes its own “device list” commands, and the framework provides a quick check:

```python
import jax
print("JAX devices:", jax.devices())
```

If these checks work, you’re in good shape for the rest of the book. If they don’t, note the error messages and driver/library versions—they will be the key clues when you (or your ops team) fix the environment. Later chapters will show more systematic environment recipes, but this quick setup checklist is enough to confirm that Python can see the hardware you plan to use.

## Exercises: measuring simple kernel performance and comparing runtimes

These exercises are meant to be quick, hands-on checks that your environment is working and that you can *see* the impact of different hardware choices.

### 1. Time a simple CPU loop

Write a tiny script that squares a large array of numbers using plain Python and then with NumPy:

```python
import time
import numpy as np

N = 10_000_000

# Pure Python list
xs = list(range(N))
t0 = time.perf_counter()
ys = [x * x for x in xs]
t1 = time.perf_counter()
print("Pure Python list comprehension:", t1 - t0, "seconds")

# NumPy array
arr = np.arange(N, dtype=np.float32)
t0 = time.perf_counter()
out = arr * arr
t1 = time.perf_counter()
print("NumPy vectorized:", t1 - t0, "seconds")
```

Compare the numbers. This is your first concrete taste of vectorization and low-level libraries doing the work.

### 2. Compare CPU vs GPU for the same kernel (if you have a GPU)

If you have PyTorch and a compatible GPU, run:

```python
import time
import torch

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda") if torch.cuda.is_available() else None
print("Using GPU:", device_gpu)

N = 10_000_000
x_cpu = torch.randn(N, device=device_cpu)

# CPU timing
t0 = time.perf_counter()
y_cpu = x_cpu * x_cpu
torch.cuda.synchronize() if torch.cuda.is_available() else None
t1 = time.perf_counter()
print("CPU time:", t1 - t0, "seconds")

if device_gpu is not None:
    x_gpu = x_cpu.to(device_gpu)

    # Warmup
    _ = x_gpu * x_gpu
    torch.cuda.synchronize()

    # GPU timing
    t0 = time.perf_counter()
    y_gpu = x_gpu * x_gpu
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print("GPU time:", t1 - t0, "seconds")
```

Note not just the raw times but also the caveats: transfer to GPU, warmup runs, and the need to call `torch.cuda.synchronize()` when timing.

### 3. Run a tiny profiler pass

Pick a small training or inference script you already have (or a simple model from a framework’s tutorial) and run the built-in profiler once.

For example, with PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

model = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
).to("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(256, 1024, device=next(model.parameters()).device)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(10):
        y = model(x)
        loss = y.sum()
        loss.backward()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
```

You don’t need to interpret every line yet. Just verify that:
- The profiler runs without crashing.
- You can see which operations dominate time on CPU vs GPU.

We’ll build on these simple measurements in later chapters, turning them into systematic profiling and benchmarking workflows. For now, the goal is to have at least one script you can run that exercises your hardware and prints back basic timing and profiling information.
