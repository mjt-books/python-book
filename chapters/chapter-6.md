# Using cuDNN, cuBLAS, and Vendor Libraries Effectively

This chapter looks under the hood of the high-performance libraries that power most GPU-accelerated Python code. You’ll build an intuition for what cuDNN, cuBLAS, and other vendor libraries do, how frameworks like PyTorch and TensorFlow use them, and which knobs you can safely tune to get better performance without writing custom CUDA.

Learning objectives:
- Explain the roles of cuBLAS, cuDNN, and similar vendor libraries in deep learning workloads.
- Recognize which high-level operations map to these libraries (GEMMs, convolutions, RNNs, etc.).
- Configure common performance-related options from Python (e.g., algorithm selection, autotuning).
- Understand trade-offs between deterministic vs fastest algorithms.
- Know when vendor libraries are “good enough” and when to consider custom kernels.

Sections:
- Why vendor libraries matter (and what problems they solve)
- cuBLAS and GEMM: matrix multiplies as the core building block
- cuDNN and convolutions: fast paths for vision and beyond
- Framework integration: how PyTorch, TensorFlow, and JAX call into vendor libraries
- Tuning knobs from Python: autotune, determinism, and workspace limits
- When vendor libraries fall short (and what to do then)
- Exercises: inspecting and nudging library usage from Python

## Why vendor libraries matter (and what problems they solve)

When you write `x @ w` or `conv2d` in Python, you are not asking Python to implement matrix multiply or convolution itself. Frameworks delegate to vendor libraries—cuBLAS, cuDNN, oneDNN, MIOpen, and others—that contain highly tuned kernels for each operation, data type, and GPU/CPU generation.

They solve three big problems for you:

- **Performance**: Implementing a fast GEMM or convolution on a modern GPU is hard. You need to pick algorithms, tile loops, use shared memory, align memory accesses, exploit tensor cores, and keep up with new hardware. Vendor libraries ship hand-optimized or auto-tuned kernels that do this far better than ad‑hoc CUDA.
- **Portability across devices**: The best algorithm for an A100 is not necessarily the best for a laptop GPU or an older data-center card. Vendor libraries choose reasonable algorithms per-device, so your high-level Python code can stay the same while performance stays “good enough” across hardware.
- **Coverage and maintenance**: Deep learning uses many combinations of shapes, strides, layouts, and dtypes (FP32, FP16, BF16, INT8). Maintaining custom kernels for all of them would be an enormous burden. Vendor libraries centralize that effort so frameworks can focus on graph building, APIs, and tooling.

For you as a Python user, the practical consequences are:

- If you can express your computation in terms of *standard ops* (GEMMs, convolutions, pooling, RNNs, etc.), you automatically benefit from these optimized kernels.
- Small changes in tensor **shape, layout, or dtype** can unlock faster code paths inside these libraries without any CUDA coding.
- A few high-level **config flags** (e.g., cuDNN benchmarking, determinism, workspace limits) give you control over the speed vs reproducibility vs memory trade-offs.

The rest of this chapter is about recognizing where vendor libraries are doing work on your behalf, how to feed them “friendly” shapes and dtypes, and which knobs you can flip from Python to get more throughput before resorting to custom kernels.

### Shape, layout, and dtype: what they mean

Because we’ll refer to these throughout the chapter, it helps to pin down a shared vocabulary:

- **Shape**: the sizes of each tensor dimension, e.g. `(batch=32, features=1024)` or `(N=32, C=3, H=224, W=224)`. Changing shape alters *how much* work there is and how well it tiles onto the hardware (e.g., many tiny matmuls vs a few large ones).
- **Layout**: the order and stride pattern of dimensions in memory, e.g. `NCHW` vs `NHWC` for images, or contiguous vs non-contiguous tensors. Layout affects whether the library can read/write memory in large, coalesced chunks or has to jump around.
- **Dtype (data type)**: how each element is represented numerically, e.g. `float32`, `float16`, `bfloat16`, `int8`. Dtype controls numerical precision and which hardware units (FP32 cores, tensor cores, etc.) and algorithms can be used.

Performance-sensitive libraries care about *all three*:

- The **shape** determines problem size and tiling opportunities.
- The **layout** determines memory access patterns and cache behavior.
- The **dtype** determines which fast paths (like tensor cores or INT8 kernels) are available.

When later sections say “this kernel is faster for certain shapes/layouts/dtypes,” they are referring back to these three separate, tunable levers.

## cuBLAS and GEMM: matrix multiplies as the core building block

If you had to pick one operation that defines modern deep learning on GPUs, it would be **GEMM**: General Matrix–Matrix Multiply. cuBLAS is NVIDIA’s highly optimized library for GEMM and related BLAS routines. Whenever you see a big dense linear layer or attention projection, there is almost certainly a GEMM—and therefore cuBLAS—under the hood.

Conceptually, a GEMM call computes:

```text
C = alpha * A @ B + beta * C
```

for matrices `A`, `B`, `C` with various layouts (row-major, column-major, transposed, etc.). Frameworks like PyTorch and TensorFlow translate many high-level ops into one or more GEMMs:

- `nn.Linear` / dense layers → one GEMM per forward (plus one for backward).
- Multi-head attention projections (Q/K/V, output projection) → multiple GEMMs.
- Some convolutions (via “im2col” or implicit GEMM) → GEMMs behind the scenes.

From Python, you rarely call cuBLAS directly; instead you write:

```python
import torch

x = torch.randn(1024, 4096, device="cuda")
w = torch.randn(4096, 2048, device="cuda")

y = x @ w  # or: torch.matmul(x, w)
```

and PyTorch maps this to an appropriate cuBLAS GEMM call for your dtypes and layouts.

A few practical implications:

- **Shape matters**: GEMM performance depends on matrix dimensions. “Tall and skinny” or “short and wide” matrices may run less efficiently than more balanced shapes. Batching (e.g., `bmm`, batched matmuls) can help keep the GPU busy.
- **Data type matters**: cuBLAS has specialized kernels for FP32, FP16/BF16, TF32, and INT8. Mixed-precision training (covered in a later chapter) is largely about letting cuBLAS and friends use faster low-precision paths where safe.
- **Layout matters**: contiguous, well-aligned tensors give GEMM implementations more room to optimize. Strided or non-contiguous layouts can introduce overhead or extra copies.

You can sometimes see evidence of cuBLAS usage in profiler traces (kernel names, library symbols), but day-to-day you mainly influence it indirectly:

- Choose tensor shapes and batch sizes that create *large* matmuls rather than many tiny ones.
- Use built-in linear/attention layers and batched matmul APIs instead of manual slices and loops.
- Let the framework’s mixed-precision utilities enable the right dtypes for you.

Later chapters (e.g., on mixed precision and quantization) will revisit GEMM from the numerical side. For now, the key idea is: if you can express more of your model as matrix multiplies, you get more for free from cuBLAS without writing any CUDA yourself.

## cuDNN and convolutions: fast paths for vision and beyond

While cuBLAS handles dense matrix multiplies, **cuDNN** is NVIDIA’s deep-learning–specific library, focused on convolutions and related operations (pooling, normalization, some RNNs). Any time you use `Conv2d` in PyTorch or a Keras `Conv2D` layer on GPU, there is almost certainly a cuDNN kernel doing the real work.

Convolutions are more complex than GEMMs because:

- They involve sliding kernels over spatial dimensions (height/width, sometimes depth and time).
- Different kernel sizes, strides, paddings, dilations, and data layouts all affect which algorithm is best.
- There are many possible algorithms (direct, FFT-based, Winograd, implicit GEMM, etc.), each with different speed and memory trade-offs.

cuDNN’s job is to:

- Implement a *family* of convolution algorithms tuned for each GPU generation and data type.
- Choose (or let the framework choose) a good algorithm for a given tensor shape, layout, and hyperparameters.
- Reuse workspace and caches to avoid re-tuning on every call.

From Python, you typically see this only through a high-level API:

```python
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1).to(device)
x = torch.randn(32, 3, 224, 224, device=device)  # NCHW: batch, channels, height, width

y = conv(x)  # under the hood: one or more cuDNN convolution kernels
```

A few practical points:

- **Layout matters**: cuDNN historically favored `NCHW` (batch, channels, height, width). Many frameworks use this layout by default on GPU. Some also support `NHWC`, especially for newer hardware and mixed-precision paths.
- **Autotuning helps**: frameworks can ask cuDNN to benchmark several convolution algorithms for a given shape and pick the fastest. In PyTorch, this is controlled by:

  ```python
  torch.backends.cudnn.benchmark = True   # allow cuDNN to autotune convs
  ```

  This can speed up repeated runs with fixed input shapes at the cost of a slower first iteration and some non-determinism.

- **Determinism vs speed**: some cuDNN algorithms are non-deterministic (e.g., due to atomic updates or parallel reductions). If you need exact reproducibility, frameworks often let you force deterministic algorithms, which may be slower.

  ```python
  torch.use_deterministic_algorithms(True)
  # or: torch.backends.cudnn.deterministic = True
  ```

- **Data type and precision**: cuDNN has optimized kernels for FP32, FP16, BF16, and INT8. Mixed-precision training (later chapter) is largely about letting cuDNN use its faster low-precision convolution paths while keeping numerics stable.

For most models, the best way to “use cuDNN well” is simple:

- Stick to standard convolutional layers with shapes and layouts your framework recommends.
- Enable autotuning when input sizes are stable and you care about throughput.
- Only tighten determinism flags when you truly need exact reproducibility, and be aware of the potential speed cost.

Later in this chapter we’ll look at how frameworks wire these libraries together and which knobs you can adjust from Python to balance speed, memory, and reproducibility for your own workloads.

## Framework integration: how PyTorch, TensorFlow, and JAX call into vendor libraries

As a Python user, you almost never call cuBLAS or cuDNN directly. Frameworks sit in front of these libraries and decide when and how to use them. Understanding that flow helps you reason about performance and what you *can* control from Python.

Very roughly, the stack looks like this:

```text
Your Python code
  ↓
Framework API (PyTorch / TF / JAX)
  ↓
Framework runtime / graph / dispatcher
  ↓
Vendor libraries (cuBLAS, cuDNN, etc.) + custom kernels
  ↓
CUDA driver + GPU
```

### PyTorch

PyTorch uses a dispatcher that maps high-level ops (e.g., `aten::matmul`, `aten::conv2d`) to backend-specific implementations. On CUDA tensors, those implementations typically:

- Prepare descriptors for cuBLAS/cuDNN (shapes, strides, dtypes, layouts).
- Choose or query an algorithm (sometimes with autotuning).
- Call the appropriate cuBLAS/cuDNN function or a custom CUDA kernel.

You control this indirectly by:

- Choosing tensor dtypes (`float32`, `float16`, `bfloat16`) and layouts.
- Enabling or disabling cuDNN benchmarking and deterministic algorithms via `torch.backends.cudnn` and `torch.use_deterministic_algorithms`.
- Structuring your model with standard layers that map cleanly onto vendor ops (Linear, Conv*, etc.).

### TensorFlow / Keras

TensorFlow builds a computation graph and uses a placement/optimization pass to decide which kernels to run where. On NVIDIA GPUs:

- Many `MatMul` and `Conv2D` nodes are lowered to cuBLAS/cuDNN calls.
- Fused ops (e.g., conv + bias + activation) often map to specialized cuDNN kernels.
- XLA (optional compiler) can further fuse and rearrange ops before hitting vendor libraries.

You influence this by:

- Using standard layers and fused ops where possible.
- Controlling XLA/JIT usage (which can change how ops are grouped and which kernels are called).
- Setting environment flags for determinism and algorithm selection when needed.

### JAX

JAX traces your Python functions into an intermediate representation (HLO) and hands them to XLA, which:

- Performs fusion and layout optimization.
- Chooses library calls (e.g., cuBLAS GEMMs) or emits custom kernels.
- Targets GPU/TPU backends.

From your side, JAX code looks like pure NumPy, but under the hood XLA is aggressively organizing work to feed vendor libraries and its own kernels efficiently.

### Why this matters to you

The details of each framework’s integration are complex, but a few practical takeaways are simple:

- If you stick to high-level, well-supported ops, you get vendor-library performance “for free.”
- Changing dtypes, shapes, or layouts can move you onto faster (or slower) code paths without any CUDA code.
- A small set of framework-level flags control big trade-offs (speed vs determinism vs memory).

Later sections in this chapter will focus on those knobs and on how to tell, via profiling and logs, whether you’re actually hitting the fast paths these libraries provide.

## Tuning knobs from Python: autotune, determinism, and workspace limits

Vendor libraries expose many options, but frameworks surface only a few that matter day to day. These knobs let you trade off **speed vs reproducibility vs memory** without touching CUDA.

### PyTorch

The main cuDNN-related toggles live under `torch.backends.cudnn`:

```python
import torch

# Let cuDNN benchmark several conv algorithms for a given shape and pick the fastest
torch.backends.cudnn.benchmark = True

# Force deterministic algorithms where possible (may be slower)
torch.backends.cudnn.deterministic = True

# Global flag for deterministic behavior (across backends)
torch.use_deterministic_algorithms(True)
```

Guidelines:

- Enable `benchmark` when:
  - Input sizes are stable (e.g., same image size and batch size every step).
  - You care about maximum throughput more than exact reproducibility.
- Enable determinism when:
  - You need bitwise-reproducible runs (e.g., debugging, research comparisons, CI).
  - You accept that some models will run noticeably slower.

For GEMMs, PyTorch and cuBLAS choose algorithms based on dtype and hardware. You influence that mostly through:

- Dtype (FP32 vs FP16/BF16 vs INT8, via AMP or quantization APIs).
- Environment variables that control math modes (e.g., TF32 usage on Ampere GPUs).

### TensorFlow / Keras

TensorFlow exposes similar ideas via environment variables and config flags. Examples (conceptual; exact APIs may evolve):

- Controlling deterministic ops:

  ```python
  import tensorflow as tf

  tf.config.experimental.enable_op_determinism()
  ```

- Enabling or disabling autotuning for certain ops is generally handled automatically, but you can inspect logs or use XLA to change fusion/placement behavior.

### JAX

JAX leans heavily on XLA, which makes its own algorithm choices. You typically “tune” via:

- Dtype choices (`jnp.float32` vs `jnp.bfloat16` etc.).
- Enabling/disabling `jit`, adjusting compilation options, or using alternative backends.

Fine-grained control over cuDNN/cuBLAS usually isn’t exposed directly, but you can influence shapes, layouts, and precision.

### Workspace and memory trade-offs

Some vendor algorithms require extra **workspace memory** to achieve peak speed. Frameworks may:

- Automatically pick algorithms within a memory budget.
- Offer flags or options to cap workspace usage (often via environment variables or advanced configs).

The trade-off is:

- More workspace → more memory use, fewer OOMs in your model, but faster kernels.
- Less workspace → more conservative algorithms, sometimes slower but more memory left for large batches/models.

For most users, the defaults are fine. You only tune workspace limits when:

- You see OOMs that go away when you reduce batch size slightly.
- Profilers/logs indicate that faster algorithms are being skipped due to workspace constraints.

In practice, your main levers from Python are:

1. Turn cuDNN benchmarking on or off.
2. Decide whether you want strict determinism.
3. Pick dtypes and shapes that unlock faster paths.

The next section discusses what to do when, even with these knobs, vendor libraries don’t give you the performance or flexibility you need.

## When vendor libraries fall short (and what to do then)

cuBLAS, cuDNN, and similar libraries cover a huge amount of ground, but not everything. Sometimes you’ll find that a critical part of your workload isn’t well served by the stock kernels. Typical symptoms:

- Profiler shows a lot of time in custom Python loops, small ops, or “fallback” kernels rather than cuBLAS/cuDNN.
- You’re using an unusual operation (custom attention variant, exotic normalization, non-standard convolution) that doesn’t map cleanly to standard primitives.
- Shapes or data layouts are so irregular that even with autotuning, performance is poor.

Before reaching for custom CUDA, there are several steps you can take:

1. **Re-express the computation in terms of supported ops.**  
   Many “custom” layers can be rewritten as a combination of matmuls, standard convolutions, and elementwise ops. If you can reduce the problem to one or two big GEMMs or convs, you get vendor-library performance almost for free.

2. **Batch and fuse at the Python level.**  
   If you see many small matmuls or convolutions, ask whether you can:
   - Use batched matmul APIs (`bmm`, `einsum`, etc.).
   - Reshape/stack tensors so multiple small operations become one larger one.
   - Replace sequences of tiny elementwise ops with a single fused expression.

3. **Try alternative layouts or dtypes.**  
   Sometimes simply changing from `NCHW` to `NHWC` (or vice versa) or enabling mixed precision moves you onto much faster kernels. This is framework- and hardware-dependent, so you need to measure.

4. **Use higher-level compiler/JIT tools.**  
   Tools like PyTorch’s `torch.compile` / TorchInductor, TensorFlow XLA, or JAX’s JIT can:
   - Fuse many small ops into larger kernels.
   - Choose better layouts automatically.
   - Generate specialized kernels for your exact workload.

   These often close the gap without any manual CUDA code.

5. **Only then consider custom kernels or extensions.**  
   If a hotspot remains after the above steps, and it truly dominates runtime, you may justify:
   - Writing a small C++/CUDA extension (PyTorch custom op, TensorFlow custom op, etc.).
   - Using kernel DSLs (Triton, TVM, etc.) to express custom GPU kernels in a more manageable way.

Custom kernels come with ongoing maintenance cost and added complexity, so they should be a last resort for clearly identified bottlenecks, not the default.

In most practical systems, getting 80–90% of the available performance comes from:

- Structuring your model and data so vendor libraries can do their job.
- Using mixed precision and autotuning wisely.
- Letting compilers/JITs fuse the rest.

The final section of this chapter will give you small exercises to practice spotting when you’re on the fast paths—and when a simple refactor is enough to get you back onto them.

## Exercises: inspecting and nudging library usage from Python

These exercises help you see vendor libraries “in action” and practice a few safe tuning tweaks from Python.

### 1. Confirm cuBLAS/cuDNN usage via a profiler

Using PyTorch and a GPU, run a tiny CNN and inspect the profiler output:

```python
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 10),
).to(device)

x = torch.randn(32, 3, 224, 224, device=device)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(10):
        y = model(x)
        loss = y.sum()
        loss.backward()
        model.zero_grad(set_to_none=True)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

You don’t need to parse every line; just note that:

- Convolution and GEMM-like ops dominate CUDA time.
- Many of those are serviced by cuDNN/cuBLAS under the hood.

### 2. Toggle cuDNN benchmarking and measure

Extend the previous script with a simple timing harness:

```python
import time, torch

def run(model, x, steps=30):
    # Warmup
    for _ in range(5):
        y = model(x)
        y.sum().backward()
        model.zero_grad(set_to_none=True)
    if x.device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        y = model(x)
        y.sum().backward()
        model.zero_grad(set_to_none=True)
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / steps

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x = x.to(device)

torch.backends.cudnn.benchmark = False
t_no_bench = run(model, x)
print("No benchmark:", t_no_bench, "s/step")

torch.backends.cudnn.benchmark = True
t_bench = run(model, x)
print("With benchmark:", t_bench, "s/step")
```

Observe whether `benchmark=True` helps for your fixed input shape. (Don’t expect huge gains on every machine, but you should see the pattern.)

### 3. Flip determinism and note the trade-off

For the same model, enable deterministic algorithms:

```python
torch.use_deterministic_algorithms(True)
# or: torch.backends.cudnn.deterministic = True

t_det = run(model, x)
print("Deterministic:", t_det, "s/step")
```

Compare:

- `No benchmark` vs `With benchmark` vs `Deterministic`.
- Note how enforcing determinism can slow things down on some hardware.

### 4. Optional: reshape computation around GEMM

Take a simple operation currently written as a Python loop over many small matmuls or linear layers, and refactor it to use a single batched matmul or a larger `nn.Linear`. Then:

1. Time both versions.
2. Run a short profiler trace on each.
3. Check how the number and size of GEMM kernels (or matmul ops) changes.

The goal is to connect a small change in how you express the computation in Python to a visible change in how vendor libraries are used under the hood and how that affects performance. You don’t need to touch CUDA directly—just write code that makes it easier for cuBLAS/cuDNN to do their job.
