# GPU Fundamentals and CUDA Basics for Python Users

This chapter introduces the essentials of GPU computing from a Python perspective. You’ll learn how GPUs differ from CPUs, what “CUDA” actually is, and how Python libraries like PyTorch and CuPy map your high-level tensor code onto low-level kernels. The goal is not to turn you into a CUDA kernel guru, but to give you enough intuition to use GPUs effectively and debug common issues.

Learning objectives:
- Understand the basic GPU execution model (threads, blocks, grids) at a conceptual level.
- Explain how data moves between host (CPU) memory and device (GPU) memory.
- Use Python libraries (e.g., PyTorch, CuPy) to run simple computations on the GPU.
- Recognize common performance pitfalls (small kernels, transfers, sync points).
- Know when and why you might drop down to custom CUDA kernels or extensions.

Sections:
- Why GPUs matter for modern AI workloads
- GPU vs CPU: architecture and execution model (intuitive view)
- Host–device memory: transfers, layouts, and pinned memory
- Basics of CUDA kernels and launch configuration (for Python users)
- Common performance pitfalls and debugging patterns
- Exercises: moving simple workloads from CPU to GPU

## Why GPUs matter for modern AI workloads

Modern deep learning is dominated by a few operations—matrix multiplies, convolutions, and pointwise kernels—applied to very large tensors. These operations are *embarrassingly parallel*: the same math runs independently on many elements. GPUs are built exactly for this pattern: they offer thousands of relatively simple cores that can execute the same instructions on different data in parallel.

From a Python user’s point of view, the impact is dramatic. A model that takes hours to train on a CPU can often be trained in minutes on a single decent GPU, and large-scale models are simply not practical without accelerators. This isn’t because your Python got faster; it’s because libraries like PyTorch or TensorFlow translate your tensor code into batched GPU kernels that keep the device’s parallel units busy.

GPUs also change the economics of experimentation. Faster training loops mean you can iterate on architectures, hyperparameters, and data processing much more quickly. That feedback loop often matters more than squeezing out the last 5% of theoretical FLOPs.

At the same time, GPUs introduce new constraints: limited device memory, explicit data transfers between host and device, and a performance profile that strongly prefers large, regular batches of work. Understanding these constraints is key to avoiding common traps like low utilization, kernel launch overhead, and spending more time on PCIe transfers than on actual compute.

In the rest of this chapter, we’ll build an intuitive model of how GPUs execute your code and how Python libraries sit on top of CUDA. The goal is not to memorize every CUDA API, but to know enough about the hardware to write GPU-friendly Python and to recognize when a performance problem is really a GPU problem versus a CPU, I/O, or data-layout problem.

## GPU vs CPU: architecture and execution model (intuitive view)

You can think of a CPU as a small team of very smart workers and a GPU as a huge crowd of simpler workers.

A **CPU** has:
- A handful of powerful cores.
- Large caches and sophisticated branch prediction.
- Great performance on code with complex control flow, lots of decisions, and relatively little data per decision.

A **GPU** has:
- Thousands of lightweight cores grouped into *warps*/*wavefronts* and *blocks*.
- Much higher memory bandwidth but higher latency to get data from main memory.
- Great performance when many threads run the *same* instructions on different data (SIMT: Single Instruction, Multiple Threads).

Conceptually, GPU work is organized as:

- **Thread**: the smallest unit of execution; runs your kernel code on one “index” of the problem.
- **Block**: a group of threads that can cooperate (e.g., via shared memory).
- **Grid**: all blocks launched for a given kernel.

When you call something like:

```python
import torch

x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
z = x @ y
```

PyTorch launches one or more CUDA *kernels* on the GPU. Each kernel is executed by many threads organized into blocks and a grid. You don’t see those details in Python, but they explain several important properties:

- GPUs want **large, regular workloads**. A big matrix multiply keeps thousands of threads busy; a tiny elementwise op on a small tensor barely lights up the device.
- There is **launch overhead** per kernel. Many tiny kernels can be slower than a few fat ones, even if the total amount of math is the same.
- Control-flow-heavy code (lots of `if`/`else` that differ between elements) can cause **divergence**: some threads in a group run one path while others idle, wasting parallelism.

CPUs and GPUs also differ in how they access memory:

- CPUs are optimized for low-latency access to a relatively small working set (deep caches).
- GPUs are optimized for high-throughput, coalesced access to large arrays; misaligned or scattered accesses hurt more.

As a Python user, you don’t need to manage threads, blocks, and warps explicitly, but it helps to remember the mental model:

- **Big, dense tensors and batched operations** map well to GPUs.
- **Lots of tiny, irregular operations** favor CPUs or require batching/fusion to perform well on GPUs.

Later in this chapter, when we talk about kernel launch configuration and performance pitfalls, we’ll keep referring back to this simple picture of “few smart CPU cores vs many simple GPU threads.”

## Host–device memory: transfers, layouts, and pinned memory

To a GPU, your data lives in two worlds:

- **Host memory**: regular system RAM, owned by the CPU.
- **Device memory**: GPU VRAM, directly accessible by GPU kernels.

Before the GPU can operate on data, that data must be in device memory. In Python libraries, this usually looks like:

```python
import torch

x_cpu = torch.randn(1024, 1024)          # host (CPU) tensor
x_gpu = x_cpu.to("cuda")                 # copy to device
y_gpu = x_gpu * 2                        # GPU kernel runs on device memory
y_cpu = y_gpu.to("cpu")                  # copy result back
```

Each `.to("cuda")` and `.to("cpu")` involves a **host–device transfer** over PCIe (or NVLink). These transfers:

- Have non-trivial latency and limited bandwidth.
- Can easily dominate runtime if overused or done with small, frequent copies.
- Often require explicit synchronization points (which can stall the GPU).

Two simple rules help a lot:

1. **Move data to the GPU once, keep it there as long as possible.**
2. **Batch transfers**: copy one big tensor rather than many tiny ones.

Memory **layout** also matters. GPUs like contiguous, well-aligned arrays. In PyTorch/NumPy:

- Prefer contiguous tensors when possible: `x = x.contiguous()` before certain ops.
- Be aware that operations like `transpose` or advanced indexing can create non-contiguous views, which some kernels handle less efficiently or with extra copies.

For input pipelines, you’ll often see the term **pinned memory** (a.k.a. page-locked memory). Pinned host memory:

- Cannot be paged out by the OS.
- Allows the GPU DMA engine to move data more efficiently.
- Can speed up host→device transfers, especially when overlapping I/O and compute.

In PyTorch’s `DataLoader`, for example:

```python
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True,   # use pinned host memory for faster transfers
)

for batch in loader:
    batch = batch.to("cuda", non_blocking=True)
    # ... run model ...
```

Using `pin_memory=True` plus `non_blocking=True` lets CUDA overlap data transfers with kernel execution when possible, improving overall throughput.

The key mental model:

- Host memory is “far” from the GPU; device memory is “close”.
- Transfers are necessary but should be **minimized, batched, and overlapped** with compute.
- Layout (contiguity, alignment) and pinned memory are quiet details that can noticeably affect how fast those transfers and kernels run.

Later in the book, when we build more complex training loops, we’ll repeatedly ask: “Where is this tensor right now—host or device—and how often are we moving it across that boundary?”

## Basics of CUDA kernels and launch configuration (for Python users)

Even if you never write a line of CUDA C++, it helps to know what a **kernel** is and what “grid” and “block” mean. That way, when you see odd performance or error messages, they’re less mysterious.

At a high level, a CUDA kernel is just a function that runs on the GPU, in parallel, across many threads. In CUDA C++ it might look like:

```cpp
// Conceptual CUDA C++ (you don't need to write this for everyday work)
__global__ void square(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global index
    if (i < n) {
        x[i] = x[i] * x[i];
    }
}
```

When launching this kernel, you choose:

- **Block size**: how many threads per block (`blockDim`).
- **Grid size**: how many blocks (`gridDim`).

Together they define how many threads run in total and therefore how the work is split.

From Python, libraries hide these details, but the same ideas apply:

- A big tensor operation (e.g., `x @ y`) launches kernels with enough threads/blocks to cover all elements.
- Many tiny operations mean many small kernel launches with overhead each time.

Some Python libraries let you write simple custom kernels without leaving Python. For example, with CuPy:

```python
import cupy as cp

# Elementwise kernel: y = x * x
square = cp.ElementwiseKernel(
    "float32 x",        # input parameters
    "float32 y",        # output parameters
    "y = x * x;",       # kernel body (per element)
    "square_kernel",    # kernel name
)

x = cp.arange(10, dtype=cp.float32)
y = square(x)
print(y)
```

CuPy picks a sensible launch configuration for you, turning the elementwise body into a full CUDA kernel.

In PyTorch, you’d typically use higher-level ops, but for very custom behavior you might:

- Use `torch.cuda` “CFFI-style” extensions, or
- Write a small C++/CUDA extension and call it from Python.

For this book, you mostly need two intuitions:

1. **Parallelism comes from many threads per kernel**. Large, regular workloads (big tensors, batched ops) give kernels enough work to scale across the GPU.
2. **Launches have overhead**. Fusing many small operations into fewer, larger ones (e.g., via built-in fused ops, JITs, or careful restructuring) can pay off even if the math is the same.

Later chapters (and vendor tools like Nsight) will show you kernel names and grid/block sizes in profiler output. You don’t have to tune them by hand, but recognizing that “this op launches thousands of tiny kernels” vs “one big kernel” will help you reason about where performance is going.

## Common performance pitfalls and debugging patterns

Once your code “runs on the GPU”, the next challenge is making sure it runs *well*. Many slow GPU programs suffer from a small set of common issues rather than anything exotic.

### 1. Too many small kernels

Symptom: GPU utilization is low, profiler shows hundreds or thousands of tiny kernels.

Causes:
- Lots of elementwise operations in Python loops.
- Frequent reshapes, indexing ops, or small custom layers.

Fixes:
- Fuse operations using built-in fused ops (e.g., `torch.nn.functional`), higher-level APIs, or JIT mechanisms where available.
- Replace Python loops over tensors with batched tensor ops.
- Profile to confirm you moved from many tiny kernels to fewer, larger ones.

### 2. Excessive host–device transfers

Symptom: Profiler/time breakdown shows a lot of time in `memcpy` or data movement; code calls `.to("cuda")` or `.cpu()` often.

Causes:
- Moving individual samples instead of whole batches.
- Copying tensors back to CPU just to compute simple metrics or logging.

Fixes:
- Move entire batches once: `batch = batch.to(device, non_blocking=True)`.
- Keep as much as possible on the GPU; only move small summaries (e.g., scalar losses) back to CPU.
- Use pinned memory + non-blocking transfers where appropriate.

### 3. Synchronization points in tight loops

Symptom: Code looks fine, but adding print statements or `.item()` calls makes it much slower.

Causes:
- CUDA is asynchronous; ops are queued. Some APIs force a **sync** (waiting for the GPU to finish), including:
  - Accessing `.item()` on a GPU tensor.
  - Moving data to CPU.
  - Some profiling/timing patterns without explicit `cuda.synchronize()`.

Fixes:
- Avoid calling `.item()` or `.cpu()` inside hot inner loops; aggregate metrics on-device and sync less often.
- When timing, explicitly synchronize once around the region you care about instead of accidentally syncing many times.

### 4. Underutilized hardware due to batch size or shapes

Symptom: GPU util hovers at low percentages even though everything is on GPU.

Causes:
- Batches are very small (e.g., `batch_size=1` for large models).
- Tensor shapes are awkward for the hardware/library algorithms.

Fixes:
- Increase batch size where memory allows, or use gradient accumulation to simulate larger batches.
- Try more “friendly” shapes if possible (e.g., multiples of 8/16/32 for channels or hidden sizes).

### 5. Input pipeline bottlenecks

Symptom: GPU utilization oscillates between high and near-zero; `DataLoader` workers are busy.

Causes:
- CPU-side data loading/augmentation too slow.
- Not enough data-loader workers or no batching/prefetching.
- Heavy Python work (e.g., image decoding) not parallelized.

Fixes:
- Increase `num_workers` in your data loader and consider `pin_memory=True`.
- Move heavy preprocessing into vectorized or native-code libraries.
- Profile the CPU side (chapter 4 + chapter 3 techniques) to see where time goes.

When something feels off, the basic debugging pattern is:

1. **Measure**: check GPU utilization (`nvidia-smi`), run a short profiler session.
2. **Identify** the dominant issue (tiny kernels, transfers, syncs, or input bottlenecks).
3. **Apply one targeted fix**, then re-measure.

You don’t need deep CUDA expertise to get large wins—just a clear mental model and a few simple checks. The exercises at the end of the chapter will walk through this loop on small, concrete examples.

## Exercises: moving simple workloads from CPU to GPU

These exercises assume you have access to a CUDA-capable GPU and a framework like PyTorch or CuPy. If not, you can still skim them for the patterns.

### 1. Basic CPU vs GPU timing for a tensor op

Using PyTorch, compare a simple elementwise operation on CPU and GPU:

```python
import time, torch

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU available:", torch.cuda.is_available())

N = 10_000_000

x_cpu = torch.randn(N, device=device_cpu)

# CPU timing
t0 = time.perf_counter()
y_cpu = x_cpu * x_cpu
t1 = time.perf_counter()
print("CPU time:", t1 - t0, "s")

if device_gpu.type == "cuda":
    # Move once, warm up, then time
    x_gpu = x_cpu.to(device_gpu)

    _ = x_gpu * x_gpu
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    y_gpu = x_gpu * x_gpu
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print("GPU time (compute only):", t1 - t0, "s")
```

Observe:

- The relative timings.
- The importance of `torch.cuda.synchronize()` when measuring.

### 2. Cost of host–device transfers

Extend the previous script to measure transfer costs:

```python
if device_gpu.type == "cuda":
    # Host -> device
    t0 = time.perf_counter()
    x_gpu = x_cpu.to(device_gpu)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print("Host->Device:", t1 - t0, "s")

    # Device -> host
    t0 = time.perf_counter()
    x_back = x_gpu.to(device_cpu)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print("Device->Host:", t1 - t0, "s")
```

Compare these times to the pure compute time. This makes it clear why you want to minimize round trips.

### 3. Moving a tiny model to GPU

Create a very small model and run one forward+backward pass on CPU vs GPU:

```python
import torch
import time

def make_model(device):
    return torch.nn.Sequential(
        torch.nn.Linear(784, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    ).to(device)

def bench(device):
    model = make_model(device)
    x = torch.randn(256, 784, device=device)
    y = torch.randint(0, 10, (256,), device=device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Warm up
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

cpu_time = bench(torch.device("cpu"))
print("CPU seconds/step:", cpu_time)

if torch.cuda.is_available():
    gpu_time = bench(torch.device("cuda"))
    print("GPU seconds/step:", gpu_time)
```

Note when the GPU helps and when overhead dominates (e.g., very tiny models).

### 4. Optional: a small CuPy port

If you use NumPy today, try porting a simple function to CuPy:

```python
import numpy as np
import cupy as cp
import time

N = 5_000_000

def cpu_version():
    x = np.random.randn(N).astype(np.float32)
    return np.tanh(x) * 2.0 + 1.0

def gpu_version():
    x = cp.random.randn(N, dtype=cp.float32)
    return cp.tanh(x) * 2.0 + 1.0

t0 = time.perf_counter()
y_cpu = cpu_version()
t1 = time.perf_counter()
print("CPU:", t1 - t0, "s")

t0 = time.perf_counter()
y_gpu = gpu_version()
cp.cuda.Stream.null.synchronize()
t1 = time.perf_counter()
print("GPU:", t1 - t0, "s")
```

Use these small experiments to build intuition: when GPUs shine, when transfers hurt, and how little code it takes in Python to switch devices while keeping the core math the same.
