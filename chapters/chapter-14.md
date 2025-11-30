# Edge Deployment: ONNX, TensorRT, and Inference on Constrained Devices

This chapter focuses on taking models that run comfortably on workstations or data-center GPUs and making them behave on edge hardware: phones, embedded boards, small GPUs, and NPUs with tight limits on memory, latency, and power. You’ll learn how to export models from common training frameworks, optimize them with ONNX- and TensorRT-based toolchains, and build lightweight inference pipelines that fit real-world constraints instead of idealized benchmarks.

Learning objectives:
- Describe common edge hardware constraints (memory, latency, power, offline operation) and how they shape model design.
- Export models from PyTorch, TensorFlow, or JAX into ONNX or other deployment formats.
- Use ONNX Runtime, TensorRT, and related tools to optimize and run models on edge devices.
- Design inference pipelines that handle preprocessing, batching, and fallback logic on constrained hardware.

Sections:
- Why edge deployment is different from server-side inference
- Edge hardware constraints and design patterns
- From training framework to ONNX (and back)
- Optimizing inference with ONNX Runtime and TensorRT
- Building and testing edge inference pipelines
- Quick setup: checking device capabilities on edge targets
- Exercises: exporting, optimizing, and benchmarking a small model

## Why edge deployment is different from server-side inference

On paper, “run this model and return a prediction” looks the same whether you are in a data center or on a tiny board zip-tied to a robot. Under the hood, the constraints are completely different. In the data center you can usually assume stable power, abundant memory, fast networking, and the ability to scale out by adding more machines. At the edge you often get the opposite: flaky power, a few gigabytes (or megabytes) of RAM, limited or no network, and strict latency budgets.

These constraints change what “good performance” means. On a server, you might optimize for throughput—how many requests per second you can handle by batching work across big GPUs. On an edge device, batching is often impossible because requests arrive one at a time from a single camera or sensor, and you still need to respond in a few milliseconds. That pushes you toward low-latency single-sample inference, aggressive quantization, and careful use of every hardware accelerator the device offers.

Failure modes also look different. In the cloud, if a process dies you restart it or route traffic elsewhere. At the edge, you might not have a backup. Your model has to coexist with other tasks (UI, sensors, control loops) on the same CPU or NPU, and a spike in memory or compute can cause missed deadlines, dropped frames, or watchdog resets instead of a clean exception.

Finally, the deployment workflows diverge. Shipping a new model to a server fleet is often automated via CI/CD, container images, and blue–green rollouts. Shipping to the edge may involve app stores, over-the-air updates with tight bandwidth, or even physically reflashing devices. That makes it more important to produce portable, self-contained artifacts (ONNX graphs, TensorRT engines, flatbuffers) that you can validate thoroughly before they ever reach a device.

Throughout this chapter, we’ll treat edge deployment not as “a smaller version of the cloud” but as its own environment with its own rules. The tools—ONNX, TensorRT, vendor runtimes—are there to help, but they work best when you design your models and pipelines with edge constraints in mind from the start.

## Edge hardware constraints and design patterns

When you deploy to edge hardware, four constraints dominate almost every decision: memory, latency, power, and connectivity. The details differ between a phone, an in-car computer, and an industrial sensor, but the themes repeat enough that we can design for them explicitly instead of treating them as unpleasant surprises.

**Memory.** Many edge devices have hundreds of megabytes of RAM available to your process, not tens of gigabytes. Large intermediate activations, oversized input resolutions, or wasteful copies that are harmless on a workstation can crash an embedded app. Common patterns here include:
- Favoring smaller backbones and width multipliers instead of “just using the big model.”
- Reducing input resolution or number of channels when possible.
- Avoiding dynamic shapes that trigger extra allocations in the runtime.
- Reusing buffers and preallocating workspaces when the runtime allows it.

**Latency.** Edge systems often serve a single stream—one camera, one sensor, one user gesture—but with a hard real-time or soft real-time budget. You care less about “throughput at batch size 128” and more about “worst-case latency for batch size 1.” This leads to patterns like:
- Designing for batch size 1 from the start (no reliance on large batches to hide kernel launch overhead).
- Preferring architectures with fewer sequential layers and cheaper operations.
- Fusing preprocessing into the model graph so you avoid Python overhead between steps.

**Power and thermals.** Phones throttle, drones run off batteries, fanless boxes overheat. Sustained 100% utilization can trigger downclocking or reboots. Design patterns here include:
- Using quantized or mixed-precision models that reduce energy per inference.
- Scheduling work in short bursts and sleeping between frames when possible.
- Choosing model sizes that meet latency targets at moderate, not peak, utilization.

**Connectivity and update cost.** Your model may need to run fully offline or handle intermittent network access. Model updates might be infrequent and expensive. This encourages:
- Packaging models and runtimes as self-contained artifacts (ONNX, TensorRT engines, mobile bundles).
- Minimizing dependencies so updates don’t require full OS or driver changes.
- Designing explicit versioning and rollback paths for model artifacts.

Putting these together, a few recurring edge design patterns emerge:

- **Tiny-but-specialized models.** Instead of shipping a single huge generalist, you deploy a small model tuned for a specific task and environment (one gesture, one scene type, one device class).
- **Cascade and fallback pipelines.** A cheap model runs first and handles most cases; a heavier model or cloud endpoint is consulted only when necessary.
- **Precompute and cache when you can.** For example, precomputing feature embeddings on the server and shipping compact classifiers to the device.

In the rest of this chapter, we’ll translate these constraints into concrete choices when exporting to ONNX, configuring ONNX Runtime or TensorRT, and wiring up the surrounding Python code. The goal is not just to make a model run on edge hardware, but to make it behave predictably within the tight box that device gives you.

## From training framework to ONNX (and back)

Most edge deployment pipelines start in a training framework (PyTorch, TensorFlow, JAX) and end in a runtime that doesn’t know anything about your original Python code. ONNX sits in the middle as a “graph exchange format”: it captures your model as a static computation graph plus weights, so other runtimes can load and optimize it without importing your training stack.

At a high level, the workflow looks like this:

1. Train and validate your model in your preferred framework.
2. Freeze the model into an inference-ready form (eval mode, fixed input shapes, no training-only layers).
3. Export to ONNX, making sure the exported opset and operators are supported by your target runtime.
4. Load the ONNX graph in a runtime (ONNX Runtime, TensorRT, or a vendor-specific engine) and verify numerics.
5. Iterate: if something breaks or performs poorly, adjust the original model and export settings.

Each framework has its own export hooks:

- **PyTorch** uses `torch.onnx.export`, which traces or symbolically executes a `forward` pass with a sample input:

  ```python
  import torch

  model = MyModel().eval()
  dummy_input = torch.randn(1, 3, 224, 224)  # batch=1, adjust to your shapes

  torch.onnx.export(
      model,
      dummy_input,
      "model.onnx",
      input_names=["input"],
      output_names=["logits"],
      opset_version=17,
      dynamic_axes=None,  # or {"input": {0: "batch"}} if you want variable batch
  )
  ```

- **TensorFlow / Keras** typically go through `tf.function` and the `tf2onnx` converter or ONNX export tools that wrap SavedModel.

- **JAX** models are often exported via intermediate formats (e.g., StableHLO) and then converted to ONNX using external tooling; in many edge scenarios, you instead export from a PyTorch/TensorFlow reimplementation of the same architecture.

Two practical considerations matter for edge targets:

- **Static shapes vs dynamic shapes.** Many edge runtimes prefer (or require) fixed input sizes. If your use case allows it, locking in a single resolution and batch size simplifies export and can unlock more aggressive optimizations.
- **Operator coverage.** Not every ONNX operator is implemented on every device. Layers that are exotic or highly framework-specific may turn into slow fallbacks—or fail to convert at all. Sticking to “boring” convolutional, linear, and activation layers usually pays off.

Going “back” from ONNX to your training framework is less common; ONNX is mostly a one-way door toward inference. In practice, you keep your source-of-truth model in the training framework, generate ONNX artifacts for deployment, and treat those artifacts as build outputs (versioned, tested, and reproducible) rather than hand-edited models. Later sections will show how to feed these ONNX files into ONNX Runtime and TensorRT, measure the resulting performance, and tune the export knobs to fit your edge device.

## Optimizing inference with ONNX Runtime and TensorRT

Once you have an ONNX model, the next step is to run it through a runtime that understands your edge hardware and can squeeze more performance out of the graph. In this section we’ll focus on two common options:

- **ONNX Runtime (ORT):** a portable engine with multiple execution providers (CPU, CUDA, TensorRT, DirectML, etc.).
- **TensorRT:** NVIDIA’s high-performance inference optimizer and runtime for their GPUs and many Jetson-class devices.

The core idea is the same for both: load the ONNX graph, apply graph-level optimizations and kernel selection, and then benchmark with realistic inputs.

### ONNX Runtime: portable baseline

ONNX Runtime is a good starting point because it runs almost everywhere and lets you swap execution providers with minimal code changes.

Basic CPU / CUDA usage:

```python
import onnxruntime as ort
import numpy as np

# Load session with desired providers
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # Enable all graph optimizations

providers = ["CPUExecutionProvider"]  # or ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession("model.onnx", sess_options, providers=providers)

# Inspect inputs/outputs
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Input:", input_name, session.get_inputs()[0].shape, session.get_inputs()[0].type)
print("Output:", output_name)

# Prepare dummy input with correct shape
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)  # Example input matching model shape

# Run inference
outputs = session.run([output_name], {input_name: dummy})  # Outputs is a list of model outputs
# outputs[0] contains the actual model output
```
*# The above code loads an ONNX model, inspects input/output details, prepares a dummy input, and runs inference. Adjust providers for your hardware.*

For edge devices, a few ORT knobs matter:

- **Execution provider choice.** On Jetson-class devices, you can combine `TensorrtExecutionProvider` and `CUDAExecutionProvider`. On CPU-only edge boards, stick to `CPUExecutionProvider` and enable all graph optimizations.
- **Optimization level.** `ORT_ENABLE_EXTENDED` or `ORT_ENABLE_ALL` enables operator fusion and constant folding that reduce latency and memory.
- **I/O binding.** On GPUs, using I/O binding to keep inputs/outputs on device can avoid host-device copies.

### TensorRT: NVIDIA-specific acceleration

TensorRT takes ONNX (or other formats) and builds a highly optimized “engine” tailored to a specific device, batch size, and precision. On NVIDIA edge hardware, this is often the fastest path to low-latency inference.

A minimal Python-side workflow looks like:

1. Use `trtexec` or the Python API to build an engine from your ONNX file.
2. Serialize the engine to disk so you don’t rebuild it at runtime.
3. Load the engine in your application and run inference with pre-allocated buffers.

Example using the `trtexec` CLI (common on Jetson):

```bash
# FP16 engine optimized for batch=1 and fixed input size
trtexec \
  --onnx=model.onnx \
  --saveEngine=model_fp16_b1.engine \
  --explicitBatch \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:1x3x224x224 \
  --maxShapes=input:1x3x224x224 \
  --fp16  # Use FP16 precision for better performance on supported hardware
```
*# This command builds a TensorRT engine from an ONNX model, optimized for FP16 precision and batch size 1.*

Key TensorRT concepts for edge:

- **Precision modes.** FP32 is the baseline; FP16 and INT8 reduce latency and memory at the cost of some accuracy. Many edge devices have tensor cores or DLA units that only light up for FP16/INT8 workloads.
- **Static shapes.** TensorRT works best with fixed shapes. Matching the shapes used at ONNX export and engine build time to your real deployment scenario is crucial.
- **Workspace size.** Engine building uses a configurable workspace; on small devices you must keep this within available memory.

You can also let ONNX Runtime *use* TensorRT under the hood by enabling the `TensorrtExecutionProvider`. This gives you a single API while still benefiting from TensorRT’s kernels where available.

### Choosing between ORT and TensorRT on edge

A practical way to decide:

- Start with **ONNX Runtime + CPU/GPU EP** to get a correct, portable baseline and simple integration.
- If you’re on NVIDIA edge hardware and need more speed or lower latency, experiment with:
  - ORT + `TensorrtExecutionProvider`.
  - Native TensorRT engines for your hottest models.

In later sections, we’ll wire these runtimes into a full edge inference pipeline, add simple benchmarking helpers, and look at how precision, batch size, and engine options impact latency and memory on real devices.

## Building and testing edge inference pipelines

So far we’ve focused on individual models and runtimes. On a real device, the model is just one stage in a pipeline that usually looks like:

1. Read input from a sensor (camera, microphone, IMU, etc.) or network.
2. Preprocess and normalize data into the model’s input format.
3. Run inference on the accelerator.
4. Postprocess outputs into decisions (labels, bounding boxes, control signals).
5. Integrate with the rest of the application (UI, logging, safety checks).

On edge hardware, every stage has to be cheap and predictable. A beautifully optimized TensorRT engine won’t save you if Python image preprocessing or JSON parsing dominates your latency.

### A minimal Python pipeline shape

Here’s a skeletal structure you can adapt for many edge scenarios:

```python
import time
import numpy as np
import onnxruntime as ort
# plus your sensor / camera library, e.g. OpenCV

def load_session(path: str) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # Enable all graph optimizations
    providers = ["CPUExecutionProvider"]  # or CUDA/TensorRT on supported devices
    return ort.InferenceSession(path, sess_options, providers=providers)

def preprocess(frame) -> np.ndarray:
    # 1. Resize/crop input frame to model's expected size
    # 2. Convert to RGB / float32
    # 3. Normalize and arrange to NCHW format
    # Keep this as vectorized NumPy/CuPy code; avoid per-pixel Python loops.
    # return array of shape (1, C, H, W)
    ...
    # Example: return np.transpose(frame, (2, 0, 1))[None].astype(np.float32)
    # Add normalization as needed

def postprocess(raw_output):
    # Map raw logits / scores / boxes to application-level decisions.
    # Keep allocations minimal; reuse buffers where possible.
    ...
    # Example: return np.argmax(raw_output, axis=1)  # For classification

def main_loop():
    session = load_session("model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Warmup: run a dummy inference to trigger runtime optimizations
    dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
    session.run([output_name], {input_name: dummy})

    while True:
        t0 = time.perf_counter()
        frame = ...  # grab from camera / sensor
        t1 = time.perf_counter()
        x = preprocess(frame)  # Preprocess input frame
        t2 = time.perf_counter()
        y = session.run([output_name], {input_name: x})[0]  # Run inference
        t3 = time.perf_counter()
        result = postprocess(y)  # Postprocess model output
        t4 = time.perf_counter()

        print(
            f"capture={t1 - t0:.3f}s "
            f"pre={t2 - t1:.3f}s infer={t3 - t2:.3f}s post={t4 - t3:.3f}s"
        )

        # Use `result` inside your application
        ...
        # Example: display result, send to actuator, etc.
```
*# This pipeline loads an ONNX model, preprocesses input, runs inference, postprocesses output, and prints timing for each stage. Replace ... with device-specific code.*

Even this toy loop captures two important edge patterns:

- **Measure each stage separately.** This tells you whether to focus on preprocessing, inference, or postprocessing first.
- **Warm up the runtime.** The first few inferences may include JIT compilation, graph optimizations, or cache misses; you don’t want those in your steady-state numbers.

### Testing pipelines on your development machine

You rarely want to debug everything directly on the smallest device you own. Instead:

- Develop and debug the full pipeline on a workstation or laptop using the *same* ONNX model and runtime API.
- Use artificial inputs (e.g., static images, prerecorded sensor logs) that you can replay consistently.
- Add simple integration tests that:
  - Load the pipeline end-to-end.
  - Run a few representative samples.
  - Assert output shapes and basic invariants (e.g., probabilities sum to ~1).

Example of a minimal “does this pipeline still work?” test:

```python
def test_pipeline_smoke():
    session = load_session("model.onnx")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Simulate a blank image
    x = preprocess(dummy_frame)  # Preprocess to model input format
    y = session.run(None, {session.get_inputs()[0].name: x})[0]  # Run inference
    assert y.ndim == 2  # Output should be 2D (batch, classes)
    assert not np.isnan(y).any()  # Check for valid outputs
```
*# This test checks that the pipeline runs end-to-end and produces valid outputs.*

Once this passes reliably on your dev machine, you copy the same code and model artifact onto the edge target and repeat the checks there, then add device-specific concerns like power draw and thermals.

### Device-specific tweaks

After you have a working pipeline, you can specialize for a given device:

- **CPU-only boards:** keep preprocessing simple and vectorized; avoid heavy Python object work inside the hot loop.
- **GPU / NPU devices:** move as much work as possible onto the accelerator (e.g., color conversion or normalization fused into the model graph).
- **Battery-powered devices:** introduce simple pacing logic (e.g., cap FPS, sleep between iterations) and measure how latency/throughput trade off against power usage.

Later sections will build on this structure when we talk about quick device capability checks and hands-on exercises for exporting, optimizing, and benchmarking models on actual edge hardware.

## Quick setup: checking device capabilities on edge targets

Before you commit to a deployment strategy, you should confirm what your edge device can actually do: which accelerators are present, how much memory you have, which precisions are supported, and whether your runtime (ONNX Runtime, TensorRT, vendor SDKs) can see them. The goal is a quick, scriptable checklist you can run on any target board.

### Basic OS-level checks

On Linux-based edge devices (Jetson, Raspberry Pi, many industrial boards), start with simple system tools:

```bash
# CPU info
lscpu | head

# Memory
free -h

# GPU / accelerator hints
lsusb || true
lspci | grep -i -E "nvidia|amd|graphics|accelerator" || true
```

On NVIDIA Jetson devices, you can also use:

```bash
# Jetson-specific summary (if available)
tegrastats  # run for a few seconds, then Ctrl+C

# GPU status
nvidia-smi || true   # works on many but not all Jetson images
```

These commands tell you roughly how much RAM you have, whether a GPU or NPU is visible to the OS, and how busy things are under load.

### Python-side capability checks

Next, probe what your inference stack sees. For ONNX Runtime:

```python
import onnxruntime as ort

print("Available execution providers:", ort.get_available_providers())  # List available backends

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
print("Session providers:", sess.get_providers())  # Show which providers are active for this session
```

On a CPU-only board you might see:

```text
Available execution providers: ['CPUExecutionProvider']
```

On an NVIDIA edge device with TensorRT properly installed you might see:

```text
Available execution providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

That tells you which backends you can realistically target from this device.

For PyTorch (useful during experimentation):

```python
import torch

print("CUDA available:", torch.cuda.is_available())  # Check if CUDA is available
print("GPU count:", torch.cuda.device_count())  # Number of GPUs detected
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))  # Name of first GPU
    print("Total GPU memory (bytes):", torch.cuda.get_device_properties(0).total_memory)  # Memory size
```

Even if your final deployment uses pure ONNX Runtime or TensorRT, PyTorch is often handy for quick “can I see the GPU and how big is it?” checks.

### TensorRT-specific sanity checks

If you plan to use TensorRT directly, verify that the CLI tools are present and can parse your ONNX model:

```bash
trtexec --onnx=model.onnx --help | head
```

Then try a tiny dry-run build (no need to keep the engine yet):

```bash
trtexec --onnx=model.onnx --explicitBatch --shapes=input:1x3x224x224 --fp16 --dryRun
```

If TensorRT can successfully parse the model, you’ll see logs about layers and supported precisions. If it fails, note the offending layer/opset—it will guide changes to your model or export settings.

### Automating a “device capability report”

It’s worth bundling these checks into a tiny script you can copy onto any new edge target:

```python
#!/usr/bin/env python3
import platform, os

def main():
    print("Platform:", platform.platform())
    print("Python:", platform.python_version())

    try:
        import onnxruntime as ort
        print("ONNX Runtime:", ort.__version__)
        print("ORT providers:", ort.get_available_providers())
    except Exception as e:
        print("ONNX Runtime not available:", e)

    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("PyTorch not available:", e)

if __name__ == "__main__":
    main()
```

Run this once per device, save the output, and you’ll have a concise description of what each edge target can support. The rest of the chapter’s recipes—export settings, runtime choices, and precision tweaks—assume you’ve done this quick capability check so you don’t aim for accelerators or features the device simply doesn’t have.

## Exercises: exporting, optimizing, and benchmarking a small model

These exercises walk through a miniature end-to-end edge deployment flow. You can complete them on a laptop first, then repeat on an actual edge device once you have one handy.

### 1. Export a small model to ONNX

Pick a simple image classifier (e.g., a tiny CNN or a pretrained backbone from your framework’s model zoo) and export it to ONNX.

Using PyTorch and a toy model:

```python
import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.relu(self.conv(x))  # Convolution + activation
        x = self.pool(x)             # Global average pooling
        x = x.view(x.size(0), -1)    # Flatten for linear layer
        return self.fc(x)            # Output logits

model = TinyCNN().eval()
dummy = torch.randn(1, 3, 64, 64)  # Example input matching model shape

torch.onnx.export(
    model,
    dummy,
    "tiny_cnn.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=17,
)
print("Exported tiny_cnn.onnx")
```
*# This code defines a small CNN, prepares a dummy input, and exports the model to ONNX format.*

Tasks:

- Inspect the exported model with `onnxruntime.InferenceSession` and confirm input/output names and shapes.
- Try changing the input resolution and see how it affects export and runtime behavior.

### 2. Run the model with ONNX Runtime and measure latency

Write a small script that loads `tiny_cnn.onnx` with ONNX Runtime and times batch-1 inference:

```python
import time
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "tiny_cnn.onnx",
    providers=["CPUExecutionProvider"],  # swap providers as appropriate
)
inp = session.get_inputs()[0]
out = session.get_outputs()[0]
print("Input:", inp.name, inp.shape, inp.type)   # Show input details
print("Output:", out.name, out.shape, out.type)  # Show output details

dummy = np.random.randn(1, 3, 64, 64).astype(np.float32)  # Example input

# Warmup: run a few inferences to trigger runtime optimizations
for _ in range(10):
    _ = session.run([out.name], {inp.name: dummy})

# Timed runs
N = 100
t0 = time.perf_counter()
for _ in range(N):
    _ = session.run([out.name], {inp.name: dummy})  # Run inference
t1 = time.perf_counter()
print(f"Avg latency: {(t1 - t0) / N * 1000:.3f} ms")  # Print average latency in ms
```
*# This script loads an ONNX model, runs warmup inferences, then measures and prints average inference latency.*

Tasks:

- Run this on your development machine and record the average latency.
- If you have an edge device, copy both the ONNX file and script there and measure again.
- Compare CPU-only vs any available accelerator providers.

### 3. Optimize with TensorRT (if on NVIDIA hardware)

If you have access to a machine or board with TensorRT, build and benchmark a TensorRT engine:

1. Use `trtexec` to build an FP16 engine:

   ```bash
   trtexec \
     --onnx=tiny_cnn.onnx \
     --saveEngine=tiny_cnn_fp16.engine \
     --explicitBatch \
     --minShapes=input:1x3x64x64 \
     --optShapes=input:1x3x64x64 \
     --maxShapes=input:1x3x64x64 \
     --fp16  # Use FP16 precision for better performance
   ```
   *# This builds a TensorRT engine optimized for FP16 precision and batch size 1.*

2. Either use `trtexec`’s built-in timing output or load the engine from Python (via the TensorRT Python API) and time batch-1 inference.

Tasks:

- Compare FP32 vs FP16 latency and memory usage, if both are available.
- Note any accuracy differences on a small validation set.

### 4. Build a tiny end-to-end edge-style loop

Re-use the “building and testing edge inference pipelines” structure, but plug in your `tiny_cnn.onnx` model:

- Simulate sensor input with randomly generated images or a small dataset folder.
- Implement `preprocess` and `postprocess` functions appropriate for your model.
- Print per-stage timings (capture, preprocess, infer, postprocess) and identify which stage dominates.

Stretch goal:

- Add a simple FPS limiter (e.g., sleep so the loop runs at ~10 Hz) and see how this interacts with CPU/GPU utilization on your device.

By the end of these exercises, you should have:

- One small model you can export reliably to ONNX.
- A baseline ONNX Runtime benchmark on at least one device.
- (Optionally) a TensorRT-optimized version and its benchmark on NVIDIA hardware.
- A minimal but realistic edge-style inference loop you can reuse in later chapters.
