# Mixed Precision, Quantization, and Numerical Trade-offs

This chapter focuses on one of the highest-leverage knobs you have for scaling AI workloads: changing how many bits you use to represent numbers. Mixed precision and quantization can dramatically speed up training and inference, shrink memory usage, and unlock specialized hardware paths—at the cost of new numerical pitfalls to watch for.

Learning objectives:
- Explain the differences between common floating-point and integer formats used in modern AI (FP32, TF32, FP16, BF16, INT8, and friends).
- Describe when mixed precision training is likely to help and how to enable it safely in common frameworks.
- Understand the basic ideas behind post-training quantization and quantization-aware training.
- Identify and mitigate common numerical issues: overflow, underflow, loss of dynamic range, and instability.
- Evaluate the trade-offs between speed, memory, accuracy, and implementation complexity when lowering precision.

Sections:
- Why precision matters for performance and memory
- Modern number formats in AI workloads
- Mixed precision training patterns in PyTorch, TensorFlow, and JAX
- Quantization for inference: PTQ, QAT, and deployment toolchains
- Numerical stability: pitfalls, debugging, and mitigation strategies
- Profiling and benchmarking precision-related speedups
- Exercises: experimenting with precision and measuring accuracy vs throughput

## Why precision matters for performance and memory

When you change precision, you are not changing your model’s *intent*—you are changing how the hardware stores and manipulates the numbers that represent that intent. That small-looking change has two huge consequences:

- How much memory each tensor consumes.
- Which hardware units and instructions your code can use.

A 32-bit float (`float32`) takes twice as many bits as a 16-bit float (`float16` or `bfloat16`). Halving the number of bits per element roughly halves the memory needed for activations, gradients, and optimizer state. On real systems this often translates to:

- Larger batch sizes before you hit “out of memory”.
- Ability to fit bigger models on the same device.
- Less data moved across PCIe, NVLink, or memory buses per step.

On GPUs and other accelerators, precision also controls *which math units you hit*. Modern hardware has specialized units (tensor cores, matrix units, dot-product engines) that are only activated for specific low-precision formats. Feed them FP16 or BF16 and you can see big throughput gains; feed them FP32 and you might fall back to slower, older-style units even on the same chip.

From a performance perspective, this gives you three main levers:

- **Throughput:** Lower precision can dramatically increase FLOPs/s for matmuls and convolutions, so each training step finishes faster.
- **Memory bandwidth:** Moving half as many bytes per element reduces pressure on memory and interconnects, letting compute units stay busier.
- **Capacity:** Fitting more of the model and activations on-device cuts down on offloading, recomputation, and checkpoint tricks.

You can see this even in a toy example. Suppose you have a model with ~1B parameters plus optimizer state, and activations that take another few GB during training:

- In FP32, you might hit the memory limit of a single GPU at a modest batch size.
- In mixed precision with FP16 activations and optimizer tricks, the same GPU may:
  - Fit a larger batch size, **or**
  - Fit a larger model at the same batch size.

When you scale training out to multiple devices, the same trade-off shows up in your network stack. Lower-precision tensors mean:

- Less data to all-reduce across devices.
- Faster parameter synchronization.
- More room to overlap communication and compute.

The catch is numerical behavior. Fewer bits means:

- Coarser spacing between representable numbers (less mantissa precision).
- Narrower or shifted dynamic ranges (different exponent layouts).
- More opportunities for underflow, overflow, or accumulation error in long chains of operations.

Mixed precision training tries to get the best of both worlds by *selectively* using low precision where it is safe (forward passes, many matmuls) and higher precision where it is not (loss scaling, gradient accumulation, certain reductions). Quantization goes further, often using integer formats (e.g., INT8) for inference-only workloads to squeeze maximum performance and minimum memory out of the hardware.

Throughout the rest of this chapter, when we talk about “turning on mixed precision” or “quantizing a model”, keep this mental model in mind:

> Fewer bits → less memory + more specialized hardware paths → higher potential speed  
> but also → less numerical headroom → more care needed in where and how you apply it.

The remaining sections make this trade-off concrete by looking at actual number formats, framework APIs, and practical debugging patterns.

## Modern number formats in AI workloads

Modern accelerators support a small zoo of numerical formats. You don’t need to memorize every bit layout, but you *do* want a feel for what each format is good at, and why your framework keeps nudging you toward one or another.

At a high level, most formats trade off three things:

- **Range:** How large or small a number you can represent before overflow/underflow.
- **Precision:** How finely you can distinguish nearby values (number of mantissa bits).
- **Hardware support:** Which units and fast paths the device can use.

Below is a practical cheatsheet you can keep in mind when choosing formats. The ranges and layouts are deliberately approximate; the goal is intuition, not bit-perfect specs.

### FP32 (float32): the baseline

- 32 bits total, with a sign bit, 8-bit exponent, and 23-bit mantissa.
- Wide dynamic range and enough precision for most deep learning tasks.
- Universally supported and typically the “safe” default.

You’ll still see FP32 used for:

- Master weights in mixed-precision training.
- Optimizer state (e.g., Adam moments).
- Reference baselines when evaluating the accuracy impact of lower precision.
- Numerically delicate parts of the computation graph (e.g., final loss computation).

If you’re not sure what precision to start with for a new model or dataset, FP32 is still the right baseline.

### TF32: GPU-friendly “almost FP32”

TensorFloat-32 (TF32) is a GPU-friendly format exposed on many recent NVIDIA GPUs:

- Uses an 8-bit exponent (like FP32) but a shorter mantissa (roughly FP16-like).
- Implemented inside tensor cores; you usually enable it via libraries (cuBLAS, cuDNN) or framework flags rather than handling it directly.
- The typical pattern is “you keep writing FP32 code, but some matmuls/convs are internally executed in TF32 on tensor cores”.

In practice:

- You write FP32 code.
- The framework may downcast certain matmuls/convolutions to TF32 internally to get tensor-core speed, while keeping FP32 elsewhere.
- Accuracy is often close enough to full FP32 for many workloads, with much higher throughput.
- For many users, “TF32 on” is a free speedup that preserves most of FP32’s stability, especially compared to jumping straight to FP16.

If you want a “safer than FP16” speed-up on supported GPUs and your framework exposes it, TF32 is often a good first switch to flip.

### FP16 (float16 / half precision)

- 16 bits: 1 sign, 5-bit exponent, 10-bit mantissa.
- Narrower range than FP32, especially for very small/very large values.
- Aggressively used on GPUs for tensor-core acceleration.

FP16 is great for:

- Forward activations and many intermediate tensors.
- Training when combined with loss scaling and FP32 master weights.
- Inference on GPUs where FP16 tensor cores are heavily optimized.

FP16 is *not* ideal for:

- Accumulating large sums (e.g., reductions over big tensors).
- Representing very small gradients without underflow.
- Layers with especially wide or heavy-tailed activation distributions.

Frameworks usually handle these pitfalls via:

- Automatic loss scaling (e.g., `GradScaler` in PyTorch).
- Performing accumulations and some reductions in FP32 even when inputs are FP16.
- Marking certain ops as “always FP32” under AMP/autocast.

A concrete example: if you sum a long vector of FP16 values that range from `1e-1` down to `1e-8`, the smallest ones may effectively vanish, because they are below the minimum representable increment at the current scale. In FP32 those tiny contributions still show up; in FP16 they are rounded away.

When you see instability under FP16 (NaNs, exploding loss, or gradients that become zero), it’s often due to this combination of smaller exponent range and fewer mantissa bits.

### BF16 (bfloat16): wider range, coarser precision

Brain Floating Point 16 (bfloat16) aims to be friendlier for training:

- 16 bits: 1 sign, 8-bit exponent, 7-bit mantissa.
- Shares FP32’s exponent width (similar *range*), but with fewer mantissa bits (less *precision*).

This means:

- Similar dynamic range to FP32, so fewer overflows/underflows than FP16.
- Less precise values, so some extra noise, but usually acceptable for deep learning.
- You often avoid the trickiest underflow issues that plague FP16 gradients.

BF16 is popular on:

- TPUs.
- Newer GPUs and some CPUs that provide BF16-optimized paths.

In many training setups, BF16 is a “safer” low-precision default than FP16: you skip a lot of loss-scaling headaches, while still getting memory and throughput gains. The trade-off is that BF16 sometimes gives slightly slower kernels than FP16 on the same device, depending on vendor support.

A pragmatic rule of thumb:

- On hardware with strong BF16 support (e.g., TPUs, newer GPUs), try BF16 first.
- Fall back to FP16 only if you need every last bit of performance and are willing to deal with more numerical tuning.

### INT8 and lower: quantization territory

Integer formats (INT8, INT4, etc.) don’t have exponents or mantissae. Instead, you choose a *scale* that maps real-valued tensors into a fixed integer range (e.g., -128 to 127). This is the domain of quantization:

- **Post-training quantization (PTQ):** Take a trained FP32/mixed model and quantize weights/activations after the fact, often with calibration data.
- **Quantization-aware training (QAT):** Simulate quantization effects during training so the model learns to be robust to them.

INT8 is especially common for:

- Inference-only workloads on servers and edge devices.
- Runtimes like TensorRT, ONNX Runtime, TFLite, and hardware NPUs.

Benefits:

- 4× smaller than FP32, 2× smaller than FP16/BF16.
- Often runs on dedicated integer dot-product or matrix units.
- Lower memory bandwidth and sometimes lower power per inference.

Costs:

- You must pick scales and zero-points carefully.
- Accuracy can drop if quantization is too aggressive or poorly calibrated.
- Debugging and tooling complexity go up compared to “just training with AMP”.

Very low-bit formats (INT4, INT2, binary) exist as research or specialized production tools. They can deliver dramatic gains but usually need heavily customized training and are much more model- and hardware-specific than INT8.

### Putting it all together

A typical modern stack might look like:

- **Training:**
  - Activations and many matmuls in FP16 or BF16.
  - Accumulations and master weights in FP32.
  - Occasional use of TF32 under the hood on supporting GPUs.

- **Inference:**
  - Offline models in FP32 or BF16 for reference and evaluation.
  - Deployed models quantized to INT8 (or lower) where latency, throughput, and memory are critical.
  - Sometimes FP16/BF16 inference as a middle ground when INT8 tooling is not mature or accuracy is too sensitive.

You can think of these formats as a ladder you climb down:

- Start in FP32 for safety and simplicity.
- Drop to TF32 or BF16 to get “easy” speedups with modest risk.
- Move to FP16 mixed precision once you’re comfortable managing loss scaling and numerical stability.
- Finally, add INT8 (or lower) quantization on the inference side when latency, cost, or device constraints force you to squeeze harder.

As you read framework docs (“AMP”, “BF16 mode”, “INT8 backend”), map those options back to this mental table: which parts are in low precision, which in FP32, and how is the hardware taking advantage of that choice?

In the next section we’ll look at concrete mixed-precision and quantization patterns in PyTorch, TensorFlow, and JAX, and how to turn these knobs without rewriting your entire training code.

## Mixed precision training patterns in PyTorch, TensorFlow, and JAX

Mixed precision training is mostly about two decisions:

1. Which tensors live in low precision vs FP32.
2. How much of that decision you let the framework automate for you.

In practice, you’ll rarely hand-pick dtypes layer by layer. Instead, you enable a framework feature (AMP, BF16 mode, etc.) and then learn the common patterns and pitfalls.

A good way to think about these APIs: they give you a “precision policy” that you can toggle in configuration, rather than hard-coding dtypes all over the model.

### PyTorch: autocast + GradScaler

PyTorch’s recommended entry point is **automatic mixed precision (AMP)** via `torch.cuda.amp` (for CUDA) or `torch.cpu.amp` (for some CPU backends):

- `autocast` decides which ops run in low precision (FP16/BF16) vs FP32.
- `GradScaler` applies dynamic loss scaling to keep FP16 gradients from underflowing.

A typical training loop:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

scaler = GradScaler()  # for FP16; keep enabled=False for pure BF16 if you like

for batch in loader:
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad(set_to_none=True)

    # Choose dtype based on config/flag
    with autocast(dtype=torch.float16):  # or torch.bfloat16
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Notes:

- The **model weights** typically stay in FP32 (“master weights”). When you call `model.to(device)` without casting, this is what happens by default.
- Forward and many matmuls/convolutions run in FP16/BF16 on tensor cores.
- Some numerically sensitive ops (e.g., softmax, batch norm reductions) stay in FP32 automatically.
- BF16 often works fine **without** loss scaling; for FP16, `GradScaler` is strongly recommended.

You don’t have to adopt AMP everywhere at once. A common migration path:

1. Implement your model and training loop in FP32.
2. Add a `precision` flag in your config or CLI.
3. Wrap your existing forward pass in `autocast` only when `precision` indicates mixed precision.
4. Enable `GradScaler` only in FP16 mode.

This keeps the diff small and makes it easy to turn mixed precision on and off during debugging.

Common failure modes and quick responses:

- Loss becomes `NaN` or `inf` immediately:
  - Check your FP32 baseline first; the model may already be unstable.
  - If FP32 is fine, try BF16 instead of FP16.
  - Lower the learning rate slightly and make sure loss scaling is on.

- Loss is finite but gradients become zero or training stalls:
  - Inspect `GradScaler`’s behavior (is it constantly reducing the scale?).
  - Try manually setting an initial scale and/or using BF16, which is less prone to gradient underflow.

- Speedup is small:
  - Profile to see if you’re input- or CPU-bound (data loading, preprocessing).
  - Check that your matmul shapes actually hit tensor cores (very tiny matmuls won’t).
  - Ensure you’re not accidentally forcing most ops back to FP32 with custom casts.

Over time, you can refine your setup with tricks like:

- Keeping embeddings or very large lookup tables in FP32 if you see accuracy regressions.
- Using AMP on some submodules but not others (via nested `autocast(enabled=False)` contexts).

### TensorFlow / Keras: global policy + `mixed_precision`

TensorFlow exposes similar capabilities via the `mixed_precision` API and global dtype policy:

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Enable mixed precision for Keras
mixed_precision.set_global_policy("mixed_float16")  # or "mixed_bfloat16"

model = build_model()            # layers will use the policy's compute dtype
optimizer = tf.keras.optimizers.Adam()

# With "mixed_float16", Keras wraps the optimizer to handle loss scaling.
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")

model.fit(dataset, epochs=10)
```

Under the hood:

- Layer *compute* dtypes become FP16/BF16.
- Variables (weights) are often kept in FP32.
- Keras adds loss scaling automatically in `mixed_float16` mode.

If you’re working with lower-level `tf.function`s, you can still control dtypes manually (casting inputs and variables), but most of the time the global policy is enough.

Things to watch:

- Ensure your input pipeline produces reasonable dtypes:
  - It’s common to store data as integers or FP32 and rely on the policy to cast to FP16/BF16 later.
- Inspect training/validation metrics against an FP32 baseline to ensure accuracy stays acceptable.
- When debugging, temporarily switch the global policy back to `"float32"`; if the issue disappears, it’s likely precision-related.

A simple pattern for experimentation:

- Start with `"float32"` and get a clean baseline.
- Flip to `"mixed_bfloat16"` on supported hardware; re-run a short training job.
- If that’s stable and gives a speedup, try `"mixed_float16"` if you want to chase more performance.

### JAX: dtype discipline and `jax.lax` control

JAX is less “magical” about mixed precision: you usually control dtypes more explicitly, but the patterns are similar:

- Choose a **compute dtype** for activations (e.g., BF16 on TPUs, FP16/TF32 on GPUs).
- Keep **accumulators** and certain reference values in FP32.
- Let XLA lower matmuls/convs to the best hardware formats.

A sketch of a training step:

```python
import jax
import jax.numpy as jnp

compute_dtype = jnp.bfloat16  # or jnp.float16
param_dtype = jnp.float32

def init_params(rng):
    params = ...  # build your param pytree in FP32
    return jax.tree_map(lambda p: p.astype(param_dtype), params)

def forward(params, x):
    x = x.astype(compute_dtype)
    # ... your model computation ...
    logits = ...
    # Often convert back to FP32 for loss and metrics
    return logits.astype(jnp.float32)

@jax.jit
def train_step(params, opt_state, batch):
    x, y = batch

    def loss_fn(p):
        logits = forward(p, x)
        loss = xent_loss(logits, y)  # computed in FP32
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state2 = optimizer.update(grads, opt_state, params)
    params2 = optax.apply_updates(params, updates)
    return params2, opt_state2, loss
```

Key habits in JAX:

- Be explicit about when you cast to/from low precision.
- Keep reductions and loss computations in FP32.
- Rely on XLA to choose good internal formats (e.g., TF32 on NVIDIA by default, BF16 on TPUs).

If you see instability when you switch to BF16/FP16, narrow it down by:

- Casting only certain submodules to low precision first.
- Using JAX’s debug/NaN checking utilities to catch where things go wrong.
- Comparing per-layer statistics (means, variances) between FP32 and low-precision runs.

### Choosing where to start

For a first, low-risk mixed precision experiment:

- **On NVIDIA GPUs:** enable PyTorch AMP or TensorFlow `mixed_float16` on an existing FP32 training script, keep everything else the same, and compare:
  - Step time / throughput.
  - Final validation metrics.
  - GPU memory usage.

- **On TPUs / BF16-capable hardware:** start with BF16 (PyTorch AMP with `dtype=torch.bfloat16`, TensorFlow `mixed_bfloat16`, or explicit JAX BF16) before FP16; it tends to “just work” for many models.

Once you’ve seen that the model trains stably and accuracy is close to FP32, you can decide whether to make mixed precision the default for that project, or keep it as a config toggle you turn on when you need more speed or capacity.

## Quantization for inference: PTQ, QAT, and deployment toolchains

Mixed precision still uses floating-point formats. Quantization goes further: it represents tensors with **integers plus a scale**, usually to make inference faster, cheaper, and more deployable on constrained hardware.

In this section we’ll stay at the “working engineer” level: enough to choose an approach, read tool docs, and interpret accuracy vs performance trade-offs.

### The basic idea: scale + integer

Most quantization schemes work like this:

- Pick a **scale** `s` (and sometimes a **zero-point** `z`).
- Map a real-valued tensor `x` to an integer tensor `q`:

  You can think of this as:

  `q = round(x / s) + z`

- At runtime, kernels operate mostly on `q` (e.g., INT8).
- When needed, you recover an approximate real value:

  `x_hat = s * (q - z)`

Choices you (or the tool) must make:

- Per-tensor vs per-channel (per-output-channel scales usually give better accuracy).
- Symmetric (`z = 0`) vs asymmetric (non-zero `z`) ranges.
- How to handle outliers and clipping.

You rarely implement this manually; instead, you configure a quantization tool and inspect its decisions.

A tiny toy example in pseudocode:

```python
# Toy symmetric per-tensor quantization to INT8
x = ...  # FP32 tensor
max_val = x.abs().max()
scale = max_val / 127.0  # map [-max_val, max_val] to [-127, 127]

q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
x_hat = (q.float() * scale)  # approximate reconstruction
```

Real frameworks do more (per-channel scales, careful calibration, better clipping strategies), but the principle is the same.

### PTQ: Post-training quantization

**Post-training quantization** takes a trained FP32/mixed model and converts it *after* training, without changing the original training loop.

Typical workflow:

1. Train and save a high-precision model.
2. Run a **calibration step** on a representative dataset:
   - Feed a few hundred or thousand samples through the model.
   - The tool records activation ranges per layer.
3. The tool chooses scales/zero-points and emits a quantized model (often INT8).

Pros:

- No changes to the training code.
- Very quick to try; great for early experiments.
- Often good enough for many vision and NLP models, especially with per-channel weight quantization.

Cons:

- Accuracy can drop noticeably for sensitive models (e.g., some generative models, small/narrow networks, regression models).
- You have less control over how the model adapts to quantization noise.

Example with PyTorch’s built-in PTQ (schematic, not a full listing):

```python
import torch
from torch.ao.quantization import quantize_dynamic

model_fp32 = load_trained_model()
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},           # which layer types to quantize
    dtype=torch.qint8,
)
torch.save(model_int8.state_dict(), "model_int8.pt")
```

For deployment, you benchmark both versions and check that the accuracy drop is acceptable given the speed/memory gains.

A small practical tip: calibration data quality matters. Using “real” production-like inputs tends to produce better scales than using a handful of random batches.

### QAT: Quantization-aware training

**Quantization-aware training (QAT)** simulates quantization during training so the model learns to be robust to it.

Core idea:

- During forward passes, weights/activations are passed through **fake quantization** nodes:
  - Values are quantized and dequantized in float, introducing the same rounding/clipping behavior as real INT8.
- Gradients still flow through (usually using straight-through estimators).
- At export time, fake-quant nodes are folded into real integer kernels.

Pros:

- Typically higher accuracy than PTQ, especially for difficult models.
- More stable behavior across hardware/toolchains, since the model saw quantization noise during training.

Cons:

- Requires changes to the training setup.
- Training becomes a bit more complex and sometimes slightly slower.
- You have yet another training variant to maintain.

Schematic PyTorch QAT sketch:

```python
import torch
from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert,
)

model = build_model()
model.train()

# Choose a backend (e.g., "fbgemm" for x86, "qnnpack" for mobile)
model.qconfig = get_default_qat_qconfig("fbgemm")
model = prepare_qat(model)  # inserts fake-quant modules

# ... train as usual for some epochs, possibly starting from a pretrained checkpoint ...

model_int8 = convert(model.eval())  # produces a quantized model for inference
```

TensorFlow and ONNX Runtime provide similar QAT flows via their tooling (e.g., TensorFlow Model Optimization Toolkit).

A common pattern is:

- Train a model in FP32 or mixed precision until convergence.
- Fine-tune with QAT for a smaller number of epochs.
- Export and deploy the quantized version.

### Deployment toolchains: ONNX, TensorRT, and friends

In real projects, you rarely handcraft quantization; you use **deployment toolchains** that:

- Parse a model (PyTorch, TF, JAX→SavedModel, etc.).
- Apply graph-level optimizations (fusion, constant folding).
- Apply PTQ/QAT-based quantization passes.
- Emit an engine or artifact tuned for a specific backend.

Common players:

- **ONNX + ONNX Runtime:**
  - Export model to ONNX.
  - Use ONNX Runtime’s quantization tools (PTQ and QAT support).
  - Run on CPU, GPU, or specialized accelerators through different execution providers.

- **TensorRT (NVIDIA GPUs):**
  - Ingests models via ONNX or framework-specific exporters.
  - Performs aggressive fusion and calibration.
  - Generates optimized FP16/INT8 engines for inference.

- **TFLite / Core ML / vendor SDKs (edge devices):**
  - Provide their own converters and quantization passes, often targeting INT8 or lower.
  - Sometimes require small “representative datasets” for calibration.

A minimal conceptual ONNX + TensorRT-style pipeline:

1. Export your FP32 or mixed-precision model to ONNX.
2. Run a quantization tool to produce an INT8 ONNX graph (PTQ or QAT-based).
3. Feed that graph into TensorRT to build an engine.
4. Benchmark latency/throughput vs FP32/FP16 baselines.

When choosing a toolchain, keep in mind:

- What devices you’re actually deploying on (CPU, GPU, mobile, custom accelerator).
- How mature the quantization support is for your target ops and architectures.
- How much extra operational complexity your team can absorb.

### When to reach for quantization

Good candidates for quantization:

- Latency-critical inference (online serving, interactive apps).
- Large-scale batch inference where hardware costs dominate.
- Edge/embedded deployments with strict memory and power budgets.

Things to watch for:

- **Accuracy regression:** always compare against a high-precision baseline on your *real* metrics, not just top-1 on a generic dataset.
- **Operational complexity:** consider whether your team can maintain the extra tooling (converters, calibration scripts, multiple model variants).
- **Debuggability:** quantized models can be harder to debug numerically; keep a simple FP32 path around for debugging and experimentation.

In the next section, we’ll look at numerical stability issues that show up when you push precision down—whether via mixed precision or quantization—and how to debug and mitigate them in practice.

## Numerical stability: pitfalls, debugging, and mitigation strategies

Lowering precision doesn’t change the *math you intend* to do, but it definitely changes the math you *actually* do. When models go unstable under mixed precision or quantization, it’s usually because small numerical issues got amplified. This section gives you a checklist for what can go wrong and what to try before giving up and going “back to FP32 everywhere”.

### Common failure modes

The symptoms tend to cluster:

- **Exploding or vanishing loss:**
  - Loss quickly becomes `inf` or `NaN`.
  - Gradients blow up in a few steps.
  - Or, conversely, loss gets stuck and stops decreasing.

- **Silent accuracy drops:**
  - Training appears stable.
  - Validation accuracy is consistently worse than FP32, even after tuning LR and epochs.
  - Certain classes or edge cases suffer more than others.

- **Brittle behavior across seeds/hardware:**
  - Some runs converge, others diverge, with only minor changes.
  - The same model behaves differently across GPUs/TPUs or quantization backends.

These usually trace back to one of a few numerical mechanisms.

### Pitfall 1: Overflow and underflow

With fewer exponent bits (FP16) or narrower integer ranges (INT8), you hit limits faster:

- **Overflow:** values exceed the maximum representable magnitude → `inf` or `NaN`.
- **Underflow:** tiny values get rounded to zero → gradients vanish before they can do useful work.

Where it shows up:

- Large intermediate activations (e.g., unbounded pre-activations before `softmax` or `exp`).
- Accumulated gradients over many steps.
- Long reduction chains (summing over big tensors).

Mitigations:

- **Loss scaling (for FP16):**
  - Scale the loss up before backpropagation, then scale gradients down.
  - Use framework-provided utilities (`GradScaler` in PyTorch, mixed precision APIs in TF).

- **Use BF16 instead of FP16 where possible:**
  - Same exponent range as FP32, so many overflow/underflow issues disappear.

- **Normalize earlier:**
  - LayerNorm/BatchNorm, residual connections, and careful initialization help keep values in a “reasonable” range.

- **Watch for obviously unbounded operations:**
  - Replace `softmax(x)` with `softmax(x - x.max(dim=...)` for numerical stability.
  - Use `logsumexp` instead of `log(softmax)` patterns.

A practical debugging trick: log the maximum absolute value of activations or gradients for a handful of layers with both FP32 and FP16/BF16 runs. If the low-precision run shows activations that spike orders of magnitude higher, overflow is a likely culprit.

### Pitfall 2: Loss of precision in reductions

Even if values stay in range, summing many low-precision values can introduce significant error:

- Dot products, means, and variances are all reductions.
- In low precision, adding a small number to a much larger one may do nothing (it gets rounded away).

Mitigations:

- **Accumulate in FP32:**
  - Most frameworks already do this for key ops.
  - When writing custom kernels or using JAX/NumPy-style code, explicitly cast accumulators to `float32`.

- **Structure computations to reduce dynamic range within a reduction:**
  - For example, subtract means before squaring for variance calculations.
  - Use numerically stable algorithms (e.g., `logsumexp`, Kahan-like summation when you implement custom reductions).

- **Be cautious with very long sequences or massive batch sizes:**
  - The more terms you add, the more rounding matters.
  - If you see subtle accuracy drift that grows with sequence length or batch size, this is a suspect.

### Pitfall 3: Sensitive layers and operations

Some parts of a model are simply more sensitive to precision than others:

- Normalization layers (BatchNorm, LayerNorm).
- Attention softmax and log-probability computations.
- Final classification layers and loss calculations.
- Some custom operations with exponentials, divisions, or subtractions of nearly equal numbers.

Mitigations:

- **Force FP32 for critical ops:**
  - Many AMP systems do this automatically, but you can override or annotate where necessary.
  - For custom layers, upcast to FP32, do the math, then downcast outputs.

- **Keep logits and losses in FP32:**
  - Cast back from FP16/BF16 to FP32 before computing cross-entropy or similar losses.
  - This reduces the risk of underflow in probabilities and improves stability of gradients.

- **In quantization:**
  - Consider leaving especially sensitive layers in higher precision (e.g., keep first/last layers FP16/FP32 while quantizing the middle).
  - Some toolchains support per-op precision decisions; take advantage of that instead of blindly quantizing the entire graph.

### Pitfall 4: Interaction with optimization and regularization

Precision issues often show up only in combination with other training choices:

- Very aggressive learning rates or schedules.
- Optimizers with complex internal state (e.g., Adam variants).
- Strong regularization or normalization that already pushes values close to numeric limits.

Things to consider:

- **Learning rate:** low precision tends to prefer slightly more conservative LR schedules. If your FP32 run is already “right on the edge,” FP16 may tip it over.
- **Optimizer state:** keep optimizer state (e.g., Adam moments) in FP32 even if activations and some weights are lower precision.
- **Regularization:** when debugging, temporarily disable or weaken regularizers (dropout, weight decay) to see if they interact badly with reduced precision.

### Debugging strategies

When something goes wrong under low precision, try to narrow down *where* it starts:

1. **Compare against an FP32 baseline.**
   - Same model, same data, same seed where possible.
   - Plot loss curves together: does mixed precision diverge early, late, or just converge to a worse value?

2. **Add runtime checks for `NaN`/`inf`.**
   - In PyTorch, you can periodically scan parameters/activations:
     ```python
     for name, p in model.named_parameters():
         if torch.isnan(p).any() or torch.isinf(p).any():
             print("Bad values in:", name)
             break
     ```
   - Or wrap suspicious layers and check their inputs/outputs.

3. **Gradual lowering of precision.**
   - Start with FP32 everywhere.
   - Then enable BF16 or FP16 for a subset of layers (e.g., only MLP blocks, not norm/attention).
   - Observe when instability appears.
   - This can quickly highlight one problematic submodule.

4. **Turn off specific optimizations.**
   - Disable certain fused kernels or TF32/INT8 paths to see if the issue is kernel-specific.
   - Many frameworks offer flags/env vars to control these features.

5. **Log statistics instead of raw tensors.**
   - Track per-layer mean, std, max, min over a few batches.
   - Look for layers where variance explodes or collapses when switching precision.
   - Compare these statistics between FP32 and low-precision runs.

6. **For quantized models, test layer-by-layer.**
   - Some tools let you run a “hybrid” model where only part of the graph is quantized.
   - Use this to identify which layers contribute most to accuracy loss.

### Practical mitigation checklist

Before abandoning mixed precision or quantization:

- Try **BF16** instead of FP16 if your hardware supports it.
- Ensure **loss scaling** is enabled and working (for FP16).
- Keep **normalization, softmax, and loss** computations in FP32.
- Accumulate **gradients and critical reductions** in FP32.
- For quantization:
  - Use **per-channel** weight quantization where available.
  - Calibrate with representative data, not just random inputs.
  - Experiment with **leaving first/last layers in higher precision**.

If, after these steps, the model still behaves poorly while an FP32 version is fine, that’s a strong signal that this particular architecture or task is unusually precision-sensitive. In those cases, you may still use mixed precision for parts of the pipeline (e.g., feature extraction, some submodules) while keeping the core model in higher precision.

In the next section, we’ll focus on systematically *measuring* the benefits of lower precision: how to profile and benchmark speedups and memory savings so you can decide, with numbers, whether the added complexity is worth it.

## Profiling and benchmarking precision-related speedups

Lower precision is a performance feature, not a fashion statement. To know whether it’s worth the added complexity, you need numbers: how much faster are training steps, how much memory did you save, and what happened to accuracy? This section gives you simple, repeatable patterns for answering those questions.

The key idea is to compare **two otherwise identical runs**:

- Same model, data, optimizer, batch size, and seed.
- Only the precision configuration changes (FP32 vs mixed precision vs quantized inference).
- You record both **speed/memory metrics** and **task metrics** (loss, accuracy, etc.).

### What to measure

For training experiments, at minimum:

- **Step time / throughput**
  - Time per training step or samples/second.
  - Measured after a short warmup (to avoid startup jitters).

- **Device memory usage**
  - Peak memory per process (e.g., via `torch.cuda.max_memory_allocated()` or framework equivalents).
  - Whether you can fit larger batch sizes.

- **Task metrics**
  - Final validation loss/accuracy or another domain-specific metric.
  - Optionally, convergence curves to see if low precision changes training dynamics.

For inference/quantization experiments:

- **Latency**
  - Time per request or per batch at the target batch size.

- **Throughput**
  - Requests per second or tokens/images per second.

- **Model size**
  - Serialized model size on disk (FP32 vs INT8, etc.).

- **Accuracy / quality**
  - Same metrics as your production evaluation.

### A minimal PyTorch timing pattern (training)

Here’s a simple helper to compare FP32 vs mixed precision training steps:

```python
import time
import torch
from torch.cuda.amp import autocast, GradScaler

def measure_step_time(model, loader, device, use_amp=False, amp_dtype=torch.float16, steps=50):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=use_amp)

    # Grab a small iterator
    it = iter(loader)

    # Warmup
    for _ in range(10):
        try:
            inputs, targets = next(it)
        except StopIteration:
            it = iter(loader)
            inputs, targets = next(it)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        ctx = autocast(dtype=amp_dtype) if use_amp else torch.cuda.amp.autocast(enabled=False)
        with ctx:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize(device)

    # Timed loop
    t0 = time.perf_counter()
    for _ in range(steps):
        try:
            inputs, targets = next(it)
        except StopIteration:
            it = iter(loader)
            inputs, targets = next(it)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        ctx = autocast(dtype=amp_dtype) if use_amp else torch.cuda.amp.autocast(enabled=False)
        with ctx:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    avg_step = (t1 - t0) / steps
    return avg_step
```

With this helper, you can run:

```python
device = "cuda"
fp32_time = measure_step_time(MyModel(), loader, device, use_amp=False)
amp_time  = measure_step_time(MyModel(), loader, device, use_amp=True, amp_dtype=torch.float16)

print("FP32 step time:", fp32_time)
print("AMP FP16 step time:", amp_time)
print("Speedup:", fp32_time / amp_time)
```

You can repeat with `amp_dtype=torch.bfloat16` to compare FP16 vs BF16.

To measure memory:

```python
torch.cuda.reset_peak_memory_stats(device)
# ... run a few training steps with your chosen precision ...
peak_bytes = torch.cuda.max_memory_allocated(device)
print("Peak memory (MB):", peak_bytes / 1024**2)
```

Run this once for FP32 and once for AMP to see how much memory you free up.

### Basic profiling hooks

Timing gives you high-level numbers; profilers tell you *why* they look that way. For precision experiments, you mainly care about:

- Whether matmuls/convolutions hit specialized low-precision kernels.
- Whether you’ve accidentally introduced extra casts or CPU bottlenecks.

In PyTorch:

```python
import torch

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(20):
        # one training or inference step under your chosen precision setup
        step()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
```

Run this once for FP32 and once for AMP; check:

- Which ops dominate CUDA time.
- Whether low-precision kernels (e.g., tensor core matmuls) appear in the AMP run but not in FP32.
- If time moved from GPU to CPU kernels (a sign of hidden bottlenecks like data loading).

Other frameworks offer similar tooling (TensorFlow Profiler, JAX’s `jax.profiler` and XLA plugin).

### Benchmarking quantized inference

For quantized models, you care mostly about *latency* and *throughput* at realistic batch sizes. A small but telling setup:

```python
import time
import torch

def benchmark_inference(model, inputs, warmup=20, runs=100):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(inputs)

        if inputs.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(inputs)
        if inputs.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    avg = (t1 - t0) / runs
    return avg

# Example: compare FP32 vs INT8 versions on the same device
inputs = sample_batch.to(device)
fp32_latency = benchmark_inference(model_fp32, inputs)
int8_latency = benchmark_inference(model_int8, inputs)

print("FP32 latency:", fp32_latency)
print("INT8 latency:", int8_latency)
print("Speedup:", fp32_latency / int8_latency)
```

Combine this with disk size measurements:

```bash
ls -lh model_fp32.pt model_int8.pt
```

You’ll get a quick view of size and speed trade-offs.

### Putting the numbers in context

Once you have basic measurements, interpret them in light of your constraints:

- If mixed precision gives you a substantial speedup and noticeable memory savings with negligible accuracy loss, it’s usually a clear win.
- If quantization yields much faster inference but costs some accuracy, decide whether that trade-off is acceptable for your application.
- If gains are small (e.g., <10%), investigate:
  - Are you I/O-bound (dataloaders, preprocessing, network)?
  - Are batch sizes too small to saturate the device?
  - Is the model too tiny to benefit from specialized hardware paths?

You don’t need a perfect benchmarking lab; a few careful scripts like these are enough to decide whether to invest more engineering time in low precision or stick with simpler, higher-precision setups.

In the final section, we’ll turn these ideas into hands-on exercises: small experiments you can run to build intuition about how precision, performance, and accuracy interact on your own hardware.

## Exercises: experimenting with precision and measuring accuracy vs throughput

These exercises are designed to be short, “change one thing and measure” experiments. The goal is not to squeeze out every last percent of performance, but to build intuition for how precision affects speed, memory, and accuracy on *your* setup.

You can treat them as standalone scripts or adapt them to an existing project.

### 1. Compare FP32 vs mixed precision training on a small model

Pick a simple vision or NLP model (e.g., a small CNN on CIFAR-10, or a tiny Transformer on a text classification task). Train it twice:

1. **Baseline FP32 run**
   - Standard training loop.
   - Record:
     - Step time / throughput (after warmup).
     - Final validation accuracy.
     - Peak GPU memory usage.

2. **Mixed precision run**
   - Enable AMP (PyTorch), `mixed_float16` (TensorFlow), or BF16 in JAX.
   - Keep everything else the same: model, optimizer, number of epochs, batch size.
   - Record the same metrics.

Then, create a tiny summary table, for example:

| Run            | Step time (s) | Speedup vs FP32 | Peak mem (GB) | Val accuracy |
|----------------|---------------|-----------------|---------------|--------------|
| FP32           |               | 1.0×            |               |              |
| AMP (FP16)     |               |                 |               |              |
| AMP (BF16)     |               |                 |               |              |

Fill in the numbers for your setup.

Questions to reflect on:

- How big is the speedup on your hardware?
- Does accuracy change in a meaningful way?
- Does BF16 behave differently from FP16, if you can test both?

### 2. Push batch size until you hit OOM (with and without mixed precision)

Using the same model, write a small loop that tries increasing batch sizes until you get an out-of-memory error:

1. Start with a modest batch size (e.g., 32).
2. Double it (64, 128, 256, …) and run a few forward/backward steps.
3. Stop when you hit a CUDA OOM (or similar) error.

Do this once for FP32 and once with mixed precision enabled. Record:

- Maximum batch size that trains without OOM in each mode.
- Peak memory usage reported by your framework (if available).

You might see something like “I can fit 2×–4× larger batches with AMP than with FP32”, which directly ties precision choice to capacity.

### 3. Quantize an inference model and measure latency/accuracy

Pick a trained inference model (e.g., a ResNet on ImageNet or a smaller classifier). Run three experiments:

1. **FP32 baseline**
   - Run inference on a fixed evaluation set.
   - Measure:
     - Per-batch or per-sample latency.
     - Overall accuracy (or your task’s main metric).

2. **FP16/BF16 inference (if supported)**
   - Enable half-precision inference (e.g., `.half()` in PyTorch on GPU, or `mixed_float16` policy in Keras).
   - Measure the same metrics.

3. **INT8 (PTQ) inference**
   - Use a PTQ tool (PyTorch, ONNX Runtime, TensorRT, TFLite, etc.) to build an INT8 version.
   - Calibrate on a small representative dataset.
   - Measure latency and accuracy again.

Summarize the results in a table:

| Precision | Model size | Latency (ms/img) | Speedup vs FP32 | Accuracy |
|-----------|------------|------------------|-----------------|----------|
| FP32      |            |                  | 1.0×            |          |
| FP16      |            |                  |                 |          |
| INT8      |            |                  |                 |          |

Questions:

- Is the accuracy drop for INT8 acceptable in your context?
- Which precision gives you the best “accuracy per millisecond”?

### 4. Stress-test numerical stability with a harder setup

Take a model that is a bit more finicky than a toy CNN—e.g., a deeper Transformer, an RNN, or a GAN—and run:

1. FP32 training as a reference.
2. FP16 mixed precision training with:
   - Default loss scaling.
   - A slightly higher learning rate than before.
3. BF16 mixed precision training (if supported).

For each run, log:

- Loss curve over time.
- Whether training ever produces `NaN` or `inf`.
- Final validation metric.

Then apply some of the mitigation strategies from the “Numerical stability” section:

- Enabling/disabling loss scaling.
- Forcing certain layers (normalization, softmax, loss) back to FP32.
- Slightly lowering the learning rate.

See which changes restore stability, and note which parts of the model are most sensitive to precision.

### 5. Optional: build a simple “precision toggle” in a config

In a small training script of your own, introduce a configuration flag like:

```python
precision = "fp32"  # or "amp_fp16", "amp_bf16"
```

Wire this flag so that:

- `precision == "fp32"` → plain FP32 training.
- `precision == "amp_fp16"` → AMP enabled with FP16.
- `precision == "amp_bf16"` → AMP enabled with BF16 (if supported).

Then, for any new model you prototype, you can:

- Run a quick FP32 baseline.
- Flip to mixed precision and immediately see the speed/memory change.
- Decide, with measurements, whether to keep it on for that project.

These exercises don’t require large clusters or long runs; they’re intentionally small and repeatable. If you save a few scripts and result tables now, you’ll have a personal reference for how your favorite models and your usual hardware respond to different precision choices—which is often more useful than any single benchmark in a paper.

Over time, this mental and empirical toolkit lets you treat “precision” as just another design dimension—alongside model architecture, optimizer choice, and data pipeline—rather than a mysterious hardware trick that you either “turn on” or “turn off”.
