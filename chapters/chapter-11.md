# TPUs and Accelerator-Specific Patterns (JAX/TPU Tips)

This chapter introduces TPUs and other highly specialized AI accelerators, focusing on how to use them from Python without getting lost in low-level details. You’ll see what makes TPUs different from GPUs, how JAX and XLA map your code to TPU hardware, and which programming patterns tend to scale cleanly across pods. The emphasis is on practical workflows: moving an existing model onto TPUs, understanding performance quirks, and deciding when an accelerator is actually the right tool for your job.

Learning objectives:
- Describe how TPUs and similar accelerators differ architecturally from CPUs and GPUs.
- Explain how JAX/XLA and related runtimes compile Python code for TPUs.
- Apply common TPU-friendly patterns for data and model parallelism.
- Diagnose basic performance issues specific to TPUs (compilation, input pipelines, sharding).
- Decide when to prefer TPUs or other accelerators over GPUs for a given workload.

Sections:
- Why TPUs and specialized accelerators exist
- TPU architecture in practice: cores, meshes, and memory
- JAX and XLA: compiling Python for accelerators
- Data and model parallelism patterns on TPUs
- Input pipelines and host–device communication
- Debugging and profiling TPU workloads
- Exercises: porting a small model to TPU and comparing throughput

## Why TPUs and specialized accelerators exist

From a distance, TPUs look like “just another fast thing to run your model on.” Underneath, they are a response to a very specific problem: deep learning workloads grew faster than general-purpose CPUs and even traditional GPUs could keep up with, both in raw compute and in cost per training run.

Most large-scale models spend the bulk of their time in a narrow set of operations: dense and sparse matrix multiplies, convolutions, normalization, and a handful of elementwise functions. Instead of building hardware that can run *any* program efficiently, TPU designers optimized for this small, predictable subset. They traded flexibility for throughput: fewer general-purpose features, more silicon devoted to wide matrix units and fast on-chip memory.

Specialized accelerators also exist because power, cooling, and datacenter footprint are not abstract concerns. If training a frontier model requires megawatts of power and racks of GPUs, the bill becomes eye-watering very quickly. TPUs and similar chips aim to deliver more useful FLOPs per joule and per dollar, especially at scale. That efficiency shows up in two places you care about: shorter time to convergence and lower cost for large experiments.

From a Python perspective, this specialization changes how you think about your code:

- You care more about regular, vectorized tensor programs that XLA can compile into large fused kernels.
- You care less about hand-writing custom CUDA ops and more about expressing your computation in a way the compiler can understand and optimize.
- You think about *compilation* time and *step* time separately, because the first few steps might be dominated by JIT compilation while later steps run at full speed.

TPUs are not the only game in town. Many vendors now ship their own AI accelerators (NPUs on laptops, PCIe cards, edge inference chips) that follow the same philosophy: narrow the workload, then build aggressive hardware for it. The high-level pattern is consistent:

1. Identify the hot path of your model (matrix-heavy, data-parallel code).
2. Express that hot path in a compiler-friendly form (JAX, TensorFlow, or another graph-style framework).
3. Let the accelerator runtime map that program onto its mesh of cores and memory.

The rest of this chapter will treat TPUs as the main example of this pattern. As you go, keep an eye out for which ideas are TPU-specific (e.g., particular mesh layouts) and which ones generalize to other accelerators (e.g., prefer fewer, larger kernel launches; minimize host–device synchronization; keep data sharded consistently). That distinction will help you reuse the same mental model when you encounter a new chip or cloud offering in the future.

## TPU architecture in practice: cores, meshes, and memory

If you look at TPU marketing slides, you’ll see terms like *core*, *chip*, *host*, *mesh*, and *pod* thrown around freely. For day-to-day Python work, you don’t need the vendor’s full block diagram, but you do need a simple picture you can map to JAX or TensorFlow concepts.

A useful mental model looks like this:

- A **TPU core** is your basic compute unit: it has a large matrix-multiply unit, some vector units, and fast scratchpad memory.
- Several cores live on a **chip**, along with high-bandwidth memory (HBM).
- Many chips are wired into a **mesh** (a grid-like interconnect) that supports fast collective operations.
- One or more **host machines** (CPUs) feed data to the mesh and orchestrate compilation.

Different TPU generations change the details—how many cores per chip, mesh dimensions, HBM size—but the programming patterns stay similar.

### Cores: matrix engines, not tiny CPUs

A TPU core is not a general-purpose CPU. It’s closer to “a big matrix engine with some helpers.” When you run a JAX function on a TPU, you’re asking XLA to turn your high-level tensor program into a sequence of big matrix and elementwise kernels that the core can execute efficiently.

Two consequences for your code:

- **Bigger is usually better.** The core likes large matrix multiplies and batched ops. Very small matmuls, many tiny kernels, or heavy Python-side control flow all underutilize the hardware.
- **Static structure wins.** Shapes and control flow that are mostly static let the compiler generate tight kernels and reuse them, instead of recompiling many variations.

You don’t directly program individual cores in Python, but you feel their preferences through performance: JAX programs that fuse into a few large ops run much faster than ones that generate hundreds of small kernels.

### Meshes: how many cores you really have

When you request “8 TPU devices” from a cloud provider, you’re really getting some slice of a larger mesh. JAX exposes these as devices:

```python
import jax

devices = jax.devices()
print(devices)
# [TPU_0, TPU_1, TPU_2, TPU_3, ...]
```

Underneath, those devices are arranged in a 2D or 3D grid (mesh). You rarely care about the *physical* layout, but you often define a **logical** mesh that matches how you want to shard your data or model. For example, you might decide:

- First mesh axis: data parallel sharding (batch dimension).
- Second mesh axis: model parallel sharding (model depth or heads).

Later, when you call `pjit` or `xmap`, you’ll tell JAX how your arrays map onto that logical mesh. The hardware’s job is to route the resulting all-reduces and all-gathers efficiently over the physical interconnect.

### Memory: on-chip SRAM, HBM, and host RAM

Like GPUs, TPUs have several memory tiers:

- **On-chip memory (SRAM)** on each core: tiny but very fast; used for tiles of matrices during compute.
- **HBM** attached to each chip: where parameters, activations, and large arrays live while you’re training.
- **Host RAM** on the CPU: where Python objects and input data typically start life.

For performance, you want hot tensors to live in HBM and stay there:

- Minimize copying arrays back and forth between host and device.
- Reuse parameters and static constants across many steps.
- Use input pipelines that stream data to the device cores in large, contiguous chunks.

From Python, you’ll mostly see this as “arrays with a device”:

```python
import jax
import jax.numpy as jnp

x = jnp.ones((1024, 1024))
print(x.device())  # TPU device if you’re running on a TPU backend
```

Moving data explicitly between host and device is rare; JAX usually handles that for you. But you *do* control how often it has to synchronize via patterns like frequent `.block_until_ready()` calls or fetching large results back to NumPy every step.

### Putting it together: how architecture shapes patterns

When you combine cores, meshes, and memory, a few practical rules emerge:

- Aim for **few, large compiled computations** instead of many small ones.
- Keep **data and parameters resident on the devices**, only crossing the host boundary when necessary.
- Use **collectives** (all-reduce, all-gather) via JAX primitives instead of manually shuffling tensors.
- Align your **batch and model dimensions** with the mesh so that sharding can be expressed cleanly.

In later sections, we’ll translate this architectural picture into concrete JAX patterns—how to define logical meshes, shard arrays, and structure your training step so that the TPU spends most of its time inside its matrix engines instead of waiting on the host or the interconnect.

## JAX and XLA: compiling Python for accelerators

So far we’ve talked about *what* TPUs look like. JAX and XLA are the pieces that decide *how* your Python code turns into something those cores can actually run. The key shift is from “Python as an interpreter that walks through each operation” to “Python as a way to *describe* a computation graph that can be compiled and shipped to accelerators.”

At a high level:

- **JAX** gives you a NumPy-like API (`jax.numpy`), automatic differentiation, and transformations like `jit`, `vmap`, and `pmap`.
- **XLA** (Accelerated Linear Algebra) is the compiler that takes the JAX computation graph, optimizes it, and emits device-specific executables for CPUs, GPUs, or TPUs.

You write pure-looking Python functions, and JAX traces those functions to build an intermediate representation that XLA can understand. XLA then handles fusion, layout, kernel selection, and mapping to the TPU mesh.

### Tracing and JIT: from Python function to compiled step

The main workhorse is `jax.jit`. It wraps a Python function and turns it into a compiled computation:

```python
import jax
import jax.numpy as jnp

def step(params, x):
    # Pure, side-effect-free tensor code
    w, b = params
    y = x @ w + b
    return jax.nn.relu(y)

# JIT-compile for the default device (CPU/GPU/TPU)
step_jit = jax.jit(step)

# First call: tracing + compilation
params = (jnp.ones((1024, 1024)), jnp.zeros((1024,)))
x = jnp.ones((256, 1024))

y = step_jit(params, x)  # triggers XLA compilation on first use
y.block_until_ready()    # ensure it actually runs before timing
```

Two important behaviors:

- **First call is expensive.** JAX traces `step`, builds a computation graph specialized to the input shapes/dtypes, and calls XLA to compile it. On TPUs this can take seconds.
- **Later calls are fast.** As long as shapes and dtypes match, JAX reuses the compiled executable and just feeds it new data.

On TPUs, this “compile once, run many times” model is crucial. You want your training step to be stable enough (no changing shapes every batch) that you compile infrequently and amortize the cost over many iterations.

### Why JAX code needs to be pure and shape-stable

Because JAX traces your function, not every Python feature plays nicely:

- Tracing sees a *symbolic* version of your arrays, not concrete values.
- Control flow that depends on array values (e.g., `if x.mean() > 0: ...`) doesn’t work the way you might expect; it gets hoisted into XLA’s control-flow ops or rejected.
- Data-dependent Python-side side effects (appending to lists, mutating globals) are invisible or problematic during tracing.

For TPU-friendly code, this leads to some best practices:

- Write **pure functions**: outputs depend only on inputs, no hidden mutation.
- Keep **shapes and dtypes mostly static** across steps; avoid dynamic Python loops over tensor dimensions inside jit-ted functions.
- Push **data-dependent control flow** into JAX primitives like `jax.lax.cond` or `jax.lax.scan` when needed.

The payoff is that XLA can see your whole step as a single, optimizable computation and generate big fused kernels that match the TPU core’s preferences.

### Devices and placement: sending work to TPUs

JAX chooses a default backend based on your environment. On a managed TPU runtime, that usually means “use the TPU.” You can inspect devices and explicitly send data or computations there:

```python
import jax
import jax.numpy as jnp

print("Available devices:", jax.devices())

tpu_devices = [d for d in jax.devices() if d.platform == "tpu"]
print("TPU devices:", tpu_devices)

# Create an array on a specific device
x = jax.device_put(jnp.ones((1024, 1024)), tpu_devices[0])
print(x.device())  # TPU device
```

For single-device TPU usage, you often don’t need to micromanage placement—`jit`-compiled functions will run on the default device, and JAX will move inputs/outputs as needed. On multi-core or multi-host TPU setups, later sections will introduce logical meshes and `pjit` for finer-grained control.

### What XLA actually buys you on TPUs

Under the hood, XLA gives several concrete benefits that align with TPU hardware:

- **Operation fusion.** Sequences like `x @ w + b` followed by `relu` get fused into a single kernel, reducing memory traffic and kernel launch overhead.
- **Layout and tiling.** XLA chooses data layouts and tiling strategies that match the TPU’s matrix units and memory hierarchy.
- **Collective optimization.** When used with parallel primitives (`pmap`, `pjit`), XLA lowers reductions and broadcasts into efficient mesh collectives.
- **Backend portability.** The same high-level JAX code can target CPU, GPU, or TPU backends with minimal changes.

From your vantage point, this means you focus on writing clean, vectorized tensor code and letting the compiler negotiate the details with the TPU. When performance is off, you’ll profile to see whether you’re compiling too often, generating too many small kernels, or forcing extra host–device transfers—rather than hand-tuning every matmul.

In the next section, we’ll build on this by introducing JAX’s data and model parallelism APIs, showing how to map that compiled step across many TPU cores and meshes without abandoning the simple “pure function + transformations” mental model.

## Data and model parallelism patterns on TPUs

Once you can compile a single training step for one TPU device, the natural next step is: how do you use *all* the cores you’re paying for? On TPUs, this usually means some combination of **data parallelism** (shard your batch) and **model parallelism** (shard your parameters or activations). JAX gives you building blocks for both, and XLA lowers them into efficient collectives on the TPU mesh.

A helpful mental model:

- Data parallelism: “each core runs a copy of the model on a different slice of the batch, then we average gradients.”
- Model parallelism: “the model itself is split across cores, and one step flows through all of them.”

You almost always start with data parallelism, because it’s simpler and works well for many workloads. Model parallelism becomes necessary when a single copy of your model no longer fits in memory on one core or device.

### Single-host data parallelism with `pmap`

The easiest entry point is `jax.pmap`, which implements SPMD (single program, multiple data) across devices on a single host. Conceptually, you write a function for *one* shard and `pmap` runs it on each device in parallel:

```python
import jax
import jax.numpy as jnp

def step(params, batch):
    # params: replicated on each device
    # batch: per-device slice of the global batch
    x, y = batch
    logits = model_apply(params, x)          # your model function
    loss = loss_fn(logits, y)
    return loss

# Wrap with pmap over a device axis called "devices"
p_step = jax.pmap(step, axis_name="devices")

# Replicate parameters across devices
params = init_params()
params_repl = jax.device_put_replicated(params, jax.devices())

# Shard a global batch across devices
def shard_batch(x):
    # Assume x.shape[0] is divisible by num_devices
    return x.reshape(jax.device_count(), -1, *x.shape[1:])

x_batch = jnp.ones((1024, 784))
y_batch = jnp.zeros((1024,), dtype=jnp.int32)
batch_sharded = (shard_batch(x_batch), shard_batch(y_batch))

losses = p_step(params_repl, batch_sharded)
```

Key TPU-friendly ideas here:

- The **global batch** is split along dimension 0 into `num_devices` shards.
- **Parameters are replicated** on each device; each core computes its own loss/gradients.
- You can use `jax.lax.pmean` inside `step` over the `axis_name` to average gradients across devices, implemented as an on-mesh all-reduce.

For many training jobs on a single host TPU slice, `pmap`-style data parallelism is enough: easy to adopt, good scaling, and relatively little new machinery to learn.

### Logical meshes and `pjit`: more flexible sharding

As models and meshes grow, you often want more explicit control than `pmap` provides. That’s where **logical meshes** and `pjit` come in. Instead of “one axis over all devices,” you define a named mesh and describe how each array is sharded across it.

A tiny example (details will vary depending on your JAX version):

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.experimental import pjit

devices = jax.devices()
mesh = Mesh(jnp.array(devices), axis_names=("data",))

@pjit.pjit(
    in_shardings=(None, NamedSharding(mesh, ("data",))),
    out_shardings=NamedSharding(mesh, ("data",)),
)
def step(params, x):
    # x is sharded over the "data" axis
    y = model_apply(params, x)
    return y
```

Compared to `pmap`:

- You can have **multiple mesh axes** (e.g., `("data", "model")`) for combined data+model parallelism.
- You can **shard parameters** as well as activations, not just replicate them.
- Sharding is specified declaratively, separate from the math in `step`.

On TPUs, `pjit` lets you express patterns like a 2D mesh where:

- One axis is used for data parallelism (bigger global batch).
- The other axis is used to shard large parameter matrices or attention heads (model parallelism).

XLA lowers the required all-gathers and all-reduces into collectives on the physical TPU mesh, so you don’t have to hand-write communication code.

### Simple vs. advanced model parallelism

Model parallelism spans a spectrum, but two patterns cover a lot of real workloads:

1. **Tensor (intra-layer) model parallelism.** Large weight matrices are split across devices. Each device holds a slice of the matrix and computes part of the output, then partial results are combined with collectives. This is common in very wide MLPs or attention layers with many heads.

2. **Pipeline (inter-layer) model parallelism.** Different layers or blocks of the model live on different devices, and a batch flows through the pipeline. This reduces per-device memory but introduces pipeline bubbles and more complex scheduling.

On TPUs with JAX, you typically start with tensor-style parallelism expressed via `pjit` and sharding rules for large arrays. Pipeline parallelism can be built on top of that, but it’s more specialized and framework-dependent, so we’ll focus on tensor-style ideas in this chapter.

A high-level guideline:

- If your model **fits** on one device but you want faster training, prefer **data parallelism** first.
- If your model **doesn’t fit** on one device (or a single host’s worth of devices), introduce **model parallelism** to shard parameters and activations.
- Combine both when you need to scale depth/width and batch size simultaneously.

### Choosing batch sizes and shard shapes

Parallelism always interacts with batch size and shapes:

- With pure data parallelism over `N` devices, you typically grow the **global batch size** so each shard’s per-device batch stays large enough to keep its core busy.
- With model parallelism, you often need to adjust layer widths, sequence lengths, or head counts so that shard sizes are balanced and map cleanly to your mesh axes.

On TPUs, where compilation is relatively heavy, it also helps to:

- Keep sharded shapes **static** across steps (no changing sequence lengths or mixture-of-experts routing patterns unless your framework handles it explicitly).
- Avoid patterns that cause many **different** compiled executables (e.g., dynamic routing that changes mesh usage from step to step).

In later sections, we’ll make these ideas concrete with a small JAX model that we first run on a single TPU, then scale up with `pmap` and finally express with a simple `pjit` sharding rule. The goal is to show how to evolve an ordinary, single-device training loop into a TPU-scale, multi-core loop without rewriting your codebase from scratch.

## Input pipelines and host–device communication

Parallelism and compilation only pay off if your TPUs have data to chew on. In many real-world TPU jobs, the limiting factor is not matmul speed but **how fast you can stream batches from storage → host CPUs → TPU cores**. A good TPU input pipeline behaves like a buffet line: always a full tray of data ready before the next core arrives.

There are three main links in the chain:

1. Reading and decoding data from disk or object storage.
2. Preprocessing and batching on the host CPUs.
3. Transferring batches to the TPU devices, ideally overlapped with compute.

If any of these stall, you’ll see TPU utilization drop even though your compiled step is efficient.

### Host-side input pipelines

On TPUs, you rarely want Python to load and preprocess each batch synchronously right before the training step. Instead, you use an input pipeline that runs *ahead* of the TPU, fills a buffer in host memory, and hands ready-made batches to JAX.

Common ingredients:

- Parallel I/O (multiple workers reading files or streams).
- Vectorized decoding and preprocessing where possible.
- Prefetching and buffering.

For example, with `tf.data` (often used even when training with JAX):

```python
import tensorflow as tf

def parse_example(serialized):
    # ... parse, decode, augment ...
    return features, labels

dataset = (
    tf.data.TFRecordDataset(tf.io.gfile.glob("gs://bucket/train-*.tfrecord"))
    .shuffle(10000)
    .repeat()
    .map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(global_batch_size, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

it = iter(dataset)
features, labels = next(it)  # tensors on the host
```

The key idea is that by the time your TPU step runs, the next batch is already sitting in host memory, ready to be transferred.

You can build similar pipelines with pure Python, `multiprocessing`, or libraries like PyTorch `DataLoader`, but on TPUs `tf.data` and cloud storage integrations are common because they’re tuned for that environment.

### Overlapping input transfer with compute

Even with a good host pipeline, you don’t want to block the TPU waiting for host→device transfers. JAX provides **prefetching** patterns that move batches to device memory one (or several) steps ahead.

A simple pattern:

```python
import jax
import jax.numpy as jnp

def prefetch_to_device(iterator, devices, prefetch_size=2):
    it = iterator

    @jax.jit
    def _prefetch(x):
        return jax.device_put_sharded(x, devices)

    # Fill an in-memory ring buffer
    buf = []
    for _ in range(prefetch_size):
        batch = next(it)
        buf.append(_prefetch(batch))

    while True:
        # Pop oldest, push newest
        batch = buf.pop(0)
        try:
            next_batch = next(it)
            buf.append(_prefetch(next_batch))
        except StopIteration:
            buf.append(None)
        yield batch
```

High-level goals:

- Move data to devices **before** the step needs it.
- Keep the TPU step itself “pure” (it just consumes in-device batches).
- Avoid extra synchronization points like frequent `.block_until_ready()` in the hot loop.

On managed TPU platforms, you’ll also see patterns where the **input pipeline runs on separate hosts** from the TPU workers, streaming data over the network. The principle is the same: keep data flowing continuously, with enough buffering to hide latency spikes.

### Minimizing host–device round-trips

Every time you pull a large tensor back from the TPU to the host, you:

- Pay transfer time over the interconnect.
- Force synchronization (the step has to finish before data is copied).

Some simple habits keep this under control:

- Log **scalars**, not full tensors. Use reductions (`mean`, `max`, etc.) on-device, then copy a small result to the host for logging.
- Avoid returning large activations or intermediate states from `jit`-compiled functions unless you really need them.
- Batch your metrics: compute them inside the training step or in a dedicated eval step that still runs on the device.

A toy example:

```python
import jax
import jax.numpy as jnp

@jax.jit
def train_step(params, batch):
    def loss_fn(p):
        logits = model_apply(p, batch["x"])
        loss = loss_fn_impl(logits, batch["y"])
        return loss, {"loss": loss, "logits": logits}

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Reduce metrics to scalars on-device
    loss_mean = jax.lax.pmean(loss, axis_name="data") if "data" in aux else loss
    return params, grads, {"loss": loss_mean}
```

Only the small `loss` scalar needs to come back to the host; everything else stays on-device and can be discarded or reused internally.

### Diagnosing input bottlenecks

The symptoms of a slow input pipeline look different from a slow TPU kernel:

- TPU utilization in monitoring dashboards is low or choppy.
- Wall-clock step time is high even though compiled FLOP/s look reasonable.
- Adding more devices doesn’t speed things up (you’re I/O-bound, not compute-bound).

To confirm, you can:

- Instrument your input pipeline to time **“data ready on host”** vs **“step finished”**.
- Run a synthetic benchmark that feeds random data directly from the host to the TPU step to see the “ideal” step time.
- Use framework-level profilers (e.g., `tf.data`’s performance tools, JAX profiling) to inspect queue depths and transfer times.

Once you see that input, not compute, is the bottleneck, the fixes are usually:

- More parallelism in reading/decoding.
- Better prefetching and caching.
- Simpler or more vectorized preprocessing.
- Fewer and larger host–device transfers per step.

In the next section, we’ll look at debugging and profiling TPU workloads more broadly—how to separate compilation time from execution time, understand collective communication costs, and verify that your carefully constructed parallelism and input pipelines are actually doing what you expect.

## Debugging and profiling TPU workloads

Debugging code that runs on TPUs feels different from debugging a local NumPy script. You can’t easily drop into a debugger *inside* a compiled step, and errors may surface only after tracing or during execution on remote devices. Profiling is also more indirect: instead of sprinkling `print` everywhere, you rely on tools that record what the compiler and devices are doing.

The goal of this section is not to cover every TPU tool in depth, but to give you a practical checklist:

- Is my time going into **compilation** or **execution**?
- Is my step limited by **compute**, **communication**, or **input**?
- When something crashes, is it a **Python-side** bug or a **device-side** one?

### Separating compilation time from step time

On TPUs, the first few calls to a `jit`/`pmap`/`pjit` function can be dominated by XLA compilation. If you time your job from scratch without warming up, you might incorrectly conclude that each step is extremely slow.

A simple pattern to separate compile vs run:

```python
import time
import jax
import jax.numpy as jnp

@jax.jit
def step(params, x):
    return model_apply(params, x)

params = init_params()
x = jnp.ones((global_batch_size, feature_dim))

# 1. Warmup + measure compile + first execution
t0 = time.perf_counter()
y = step(params, x)
y.block_until_ready()
t1 = time.perf_counter()
print("Compile + first step:", t1 - t0, "s")

# 2. Measure steady-state step time
num_steps = 10
t0 = time.perf_counter()
for _ in range(num_steps):
    y = step(params, x)
y.block_until_ready()
t1 = time.perf_counter()
print("Average step time (no compile):", (t1 - t0) / num_steps, "s")
```

If “compile + first step” is huge and “average step time” is small, your kernel is fine and you need to:

- Avoid patterns that force frequent recompilation (changing shapes, Python branching on tensor values).
- Reuse compiled functions across epochs and jobs when possible.

If both numbers are large, the problem is in the compiled step itself or in input/communication.

### Using JAX profiling hooks

JAX integrates with backend profilers so you can see where time is going on the TPU. In many environments you can use a context manager to capture a trace:

```python
from jax.experimental import jax2tf  # just an example import; adjust per JAX version
import jax

with jax.profiler.trace("profiles/tpu_trace"):
    for _ in range(50):
        y = step(params, x)
    y.block_until_ready()
```

You can then load the resulting trace (often a directory of events) into tools such as:

- TensorBoard’s **Profile** plugin.
- Chrome Trace Viewer.
- Cloud-specific TPU profiling UIs.

What you look for:

- Long **Compile** events vs short **Execute** events → too much compilation.
- Large gaps between executes → likely input stalls or host-side work.
- Heavy **collective** operations (all-reduce/all-gather) → communication-bound, maybe over-sharded.

Exact APIs and tooling evolve, but the pattern is stable: wrap a representative training window in a profiling context, then inspect the trace for time-dominant ops and idle periods.

### Printing and debugging inside JIT

Debugging TPU code by printing is tricky because `jit` and `pmap` trace your function once, then run a compiled version many times. Plain `print` in a jitted function may run only at trace time (with abstract values) or be removed entirely.

JAX provides `jax.debug.print` (or similar, depending on version), which is designed to work under JIT:

```python
import jax
import jax.numpy as jnp

@jax.jit
def step(params, x):
    y = model_apply(params, x)
    jax.debug.print("Batch mean: {m}", m=jnp.mean(y))
    return y
```

Guidelines:

- Use debug printing sparingly—it can slow your step and clutter logs, especially under `pmap`.
- Prefer logging small summaries (means, norms, min/max) instead of full tensors.
- Turn debug prints off once you’ve isolated the issue; don’t leave them in hot paths of production training code.

For logic bugs that happen **before** JIT (e.g., wrong shapes, bad batch sharding), it’s often easier to:

- Run a CPU backend (`jax_default_device=cpu`) with small toy inputs.
- Disable JIT entirely (`jax.config.update("jax_disable_jit", True)`) to get normal Python stack traces, then re-enable JIT once fixed.

### Common TPU-specific failure modes

Some issues show up more often on TPUs than on local CPUs/GPUs:

- **Out-of-memory on device.** You’ll see OOM errors during compilation or execution when weights/activations/temporary buffers don’t fit into HBM. Fixes: smaller batch size, activation checkpointing, model sharding, or reducing precision.

- **Shape or sharding mismatches under `pmap`/`pjit`.** Errors complaining about inconsistent shard sizes, non-divisible batch dimensions, or incompatible mesh shapes. These usually trace back to:
  - Global batch sizes not divisible by device counts.
  - Arrays that are replicated when you thought they were sharded (or vice versa).
  - Inconsistent sharding annotations between parameters and activations.

- **Host-side timeouts or stalls.** If input pipelines are slow or Python-side code does heavy work inside a jitted function (e.g., large Python loops, complex control flow), you may see long gaps between TPU executions or watchdog timeouts.

The debugging loop is:

1. Reproduce with **small inputs** and fewer devices if possible.
2. Turn off JIT or run on CPU/GPU to get clearer stack traces.
3. Re-enable JIT and profile to see where the time is going.
4. Only then try TPU-specific fixes (sharding changes, layout tweaking).

### Putting profiling into your regular workflow

You don’t need to profile every experiment, but you should have a lightweight habit:

- When moving a model to TPUs for the first time, run:
  - A small warmup.
  - A short profiled run (e.g., 50–100 steps).
- Save:
  - Compile time vs steady-state step time.
  - A trace file you can open later if performance regresses.
- When you change **parallelism**, **input pipeline**, or **precision**, repeat the same short profiling ritual.

Over time, you’ll build intuition for what “normal” looks like for your models on TPUs. That makes it much easier to spot when a new commit or configuration pushes you into recompile hell, I/O starvation, or communication bottlenecks—long before you waste hours on a full training run.

In the final section of this chapter, we’ll put everything together in a small, end-to-end exercise: porting a compact JAX model to TPUs, measuring compile and step times, and experimenting with batch size and sharding to see how throughput and utilization change in practice.

## Exercises: porting a small model to TPU and comparing throughput

These exercises are meant to turn the chapter’s ideas into something concrete you can run. The focus is not on achieving perfect TPU utilization, but on *seeing* how compilation, device placement, and parallelism change behavior and throughput.

### 1. Run the same JAX model on CPU, GPU, and TPU (if available)

Start with a small MLP in JAX that you can run on any backend:

```python
import time
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)

def init_mlp(key, in_dim=784, hidden=1024, out_dim=10):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    params = {
        "w1": jax.random.normal(k1, (in_dim, hidden)) * 0.01,
        "b1": jnp.zeros((hidden,)),
        "w2": jax.random.normal(k2, (hidden, hidden)) * 0.01,
        "b2": jnp.zeros((hidden,)),
        "w3": jax.random.normal(k3, (hidden, out_dim)) * 0.01,
        "b3": jnp.zeros((out_dim,)),
    }
    return params

def mlp_apply(params, x):
    x = jnp.dot(x, params["w1"]) + params["b1"]
    x = jax.nn.relu(x)
    x = jnp.dot(x, params["w2"]) + params["b2"]
    x = jax.nn.relu(x)
    x = jnp.dot(x, params["w3"]) + params["b3"]
    return x

@jax.jit
def forward(params, x):
    return mlp_apply(params, x)

batch_size = 1024
x = jax.random.normal(key, (batch_size, 784))
params = init_mlp(key)

def time_forward(device):
    print(f"\nRunning on device: {device}")
    x_d = jax.device_put(x, device=device)
    params_d = jax.device_put(params, device=device)

    # Compile + first run
    t0 = time.perf_counter()
    y = forward(params_d, x_d)
    y.block_until_ready()
    t1 = time.perf_counter()
    print("Compile + first step:", t1 - t0, "s")

    # Steady-state
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        y = forward(params_d, x_d)
    y.block_until_ready()
    t1 = time.perf_counter()
    print("Avg step (no compile):", (t1 - t0) / iters, "s")

for d in jax.devices():
    time_forward(d)
```

Tasks:

- Run this script on:
  - A CPU-only machine.
  - A GPU machine (if available).
  - A TPU runtime (if available).
- Record:
  - Compile + first step time.
  - Average step time per backend.
- Note how much larger the “compile” penalty is on TPU vs CPU/GPU, and how the steady-state times compare.

### 2. Add data parallelism with `pmap` on a multi-core TPU

Extend the previous example to use `pmap` over available TPU devices:

```python
import jax
import jax.numpy as jnp

num_devices = jax.device_count()
print("Devices:", jax.devices())

def make_batch(batch_size_per_device, in_dim=784):
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (num_devices * batch_size_per_device, in_dim))
    y = jax.random.randint(key, (num_devices * batch_size_per_device,), 0, 10)
    return x, y

def loss_fn(params, x, y):
    logits = mlp_apply(params, x)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    loss = jnp.mean(jnp.sum((logits - one_hot) ** 2, axis=-1))
    return loss

def shard_batch(x):
    return x.reshape(num_devices, -1, *x.shape[1:])

@jax.pmap
def p_forward(params, batch):
    x, y = batch
    return loss_fn(params, x, y)

params = init_mlp(jax.random.PRNGKey(0))
params_repl = jax.device_put_replicated(params, jax.devices())

batch_size_per_device = 256
x, y = make_batch(batch_size_per_device)
batch_sharded = (shard_batch(x), shard_batch(y))
```

Tasks:

- Time `p_forward(params_repl, batch_sharded)` (compile + steady-state) similarly to the single-device version.
- Increase `batch_size_per_device` and observe:
  - When does throughput (examples/sec) improve?
  - When does you hit diminishing returns or OOM?

### 3. Experiment with input pipeline and host–device overlap

Replace the synthetic `make_batch` with a simple input iterator (e.g., data from disk or randomly generated on the host), and wrap it with a prefetcher similar to the one in this chapter.

Tasks:

- Measure step time:
  - Without prefetching (synchronous host→device transfer each step).
  - With a small prefetch buffer (e.g., 2–4 steps).
- Use profiling tools in your environment to confirm:
  - Fewer idle gaps between executes when prefetching is enabled.
  - Lower contribution of input pipeline to total step time.

### 4. Optional: profile and inspect a short TPU run

Pick one of the above TPU runs and wrap 50–100 steps in a JAX profiling context (or the TPU profiler your environment provides). Open the trace and answer:

- How much of the time is compilation vs execution?
- Are there obvious input stalls (gaps) or very large collective ops?
- Does changing batch size or the number of devices shift the bottleneck?

These exercises don’t require a large model or a full training loop. The aim is to build intuition for how TPUs behave compared to CPU/GPU, and to practice the workflow you’ll use on real projects: port → sanity check → profile → adjust batch/sharding/input → re-profile.
