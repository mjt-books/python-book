# Energy Efficiency and Cost-Aware Training Strategies

This chapter focuses on making your AI workloads not just fast, but *economical*—in terms of cloud spend, on-prem capacity, and energy use. You will learn how to measure and reduce the cost per experiment, choose hardware with price/performance in mind, and design training and inference setups that respect both budget and power constraints without sacrificing reliability.

Learning objectives:
- Define and measure key cost and efficiency metrics (e.g., cost per training run, cost per 1,000 inferences, energy per experiment).
- Compare hardware options in terms of price/performance and energy efficiency for different workloads.
- Apply model- and system-level techniques (batching, scheduling, early stopping, quantization) to reduce cost and power draw.
- Build simple monitoring loops that track utilization, spend, and energy-related signals during training and inference.

Sections:
- Why energy and cost matter for AI workloads
- Metrics: utilization, throughput, cost, and energy
- Hardware choices and price/performance trade-offs
- Software-level levers: batching, precision, and scheduling
- Optimizing training workflows for cost and energy
- Optimizing inference and serving pipelines
- Monitoring, budgeting, and guardrails
- Exercises: estimating and reducing cost per experiment

## Why energy and cost matter for AI workloads

If you train a small model once on a free-tier GPU, your energy use and cloud bill barely register. As soon as you run dozens of experiments, keep clusters busy around the clock, or ship a model that serves millions of requests per day, the story changes. What felt like “just a few dollars” or “just one GPU” turns into a recurring line item on your budget and a non-trivial contributor to your organization’s energy footprint.

From a technical perspective, most AI workloads are simple: push tensors through layers, compute gradients, update weights, repeat. What makes them expensive is how long they run, how heavily they use specialized hardware, and how much idle time or waste creeps in. A cluster full of half-utilized GPUs burns almost as much power—and money—as one running at 90% utilization, but delivers far less useful work.

Cost and energy also show up as invisible constraints on your experiments. If each training run is expensive, you run fewer of them. That means less hyperparameter search, fewer ablation studies, and slower iteration cycles. In practice, this can matter more than raw model quality: a team that can cheaply run many experiments often outperforms a team with fancier hardware but a tighter budget.

There is a broader context too. Large-scale AI systems consume significant electricity, and that electricity is not always clean. Many organizations now track emissions, data-center efficiency, and energy use per unit of work. Even if your primary goal is “ship the model”, designing for efficiency—choosing batch sizes, schedules, and hardware with cost and power in mind—aligns with both budget realities and sustainability targets.

The good news is that you usually don’t need entirely new tools to improve efficiency. The same knobs you already tune for performance—batch size, precision, parallelism, scheduling—also control how much energy and money each unit of progress costs. The focus in this chapter is to make those trade-offs explicit: instead of only asking “how fast is this?”, we’ll ask “how much useful work per joule, per dollar, and per hour of wall-clock time do we get?” Once you start thinking in those terms, many everyday design decisions naturally shift toward more efficient patterns.

## Metrics: utilization, throughput, cost, and energy

You can’t tune what you don’t measure. Most teams look at wall-clock time (“this run took 6 hours”) and maybe a monthly cloud bill, but that’s not enough to guide everyday decisions. To reason about efficiency, you want a small set of metrics that connect *hardware behavior* to *money and energy* in a way you can automate and compare across runs.

At the hardware level, two concepts show up everywhere:

- **Utilization**: how busy a device is over time.
- **Throughput**: how much useful work it completes per unit time.

For GPUs and accelerators, utilization is often reported as a percentage over short intervals (for example, via `nvidia-smi` or framework-specific profilers). A GPU that hovers around 30% utilization during training is a red flag: you are paying for peak performance but only using a fraction of it, often because of slow data loading, small batches, or frequent synchronization points.

Throughput is usually expressed in domain-specific units:

- For training: *examples per second*, *tokens per second*, or *steps per hour*.
- For inference: *requests per second* (RPS), possibly broken down by model or endpoint.

A simple training-loop metric might look like:

```python
# ...existing code to set up model, optimizer, dataloader...
import time

start = time.perf_counter()
num_examples = 0

for step, batch in enumerate(dataloader, start=1):
    # ...forward, loss, backward, optimizer step...
    num_examples += batch["inputs"].shape[0]

    if step % 100 == 0:
        elapsed = time.perf_counter() - start
        throughput = num_examples / elapsed
        print(f"[step {step}] {throughput:.1f} examples/sec")
```

On its own, throughput tells you how *fast* the run is. To make it cost-aware, you just divide by a price:

- If your instance costs `$X` per hour and your job runs for `T` hours, the *run cost* is `X * T`.
- If that run processes `N` examples, then **cost per example** is `(X * T) / N`.
- For inference, **cost per 1,000 requests** is a convenient unit: `(X * T * 1000) / R`, where `R` is the number of requests served during `T`.

Similarly, if you know or approximate power usage (for example, from `nvidia-smi --query-gpu=power.draw` or cloud provider estimates), you can compute **energy per run** and **energy per example**:

- Energy for a device: `average_power_watts * duration_hours`.
- Energy per 1,000 inferences: `(energy_for_run * 1000) / num_requests`.

You don’t need perfect measurements to get value. Even rough, consistently collected metrics let you compare:

- Configuration A vs B (for example, different batch sizes).
- Hardware A vs B (for example, different GPU types).
- Version N vs N+1 of your model or data pipeline.

The key habit is to turn “this felt faster” into a simple record like:

- `throughput_examples_per_sec`
- `gpu_utilization_percent`
- `cost_per_1k_examples`
- `energy_kwh_per_1k_examples` (even if initially estimated)

Later sections will show how to automate this collection so that every experiment and deployment produces comparable, cost- and energy-aware metrics by default.

## Hardware choices and price/performance trade-offs

Once you can measure throughput and cost, hardware decisions stop being abstract and start looking like simple ratios. Instead of arguing about “fast vs cheap” in the abstract, you can ask: *how much useful work do we get per dollar, per joule, and per hour on each option?*

At a high level:

- **CPUs** are flexible and often already paid for (on-prem), but usually have lower peak throughput for dense tensor math.
- **GPUs** and **accelerators** cost more per hour and draw more power, but can deliver far more work in that hour—if you keep them busy.
- **Older or smaller devices** may be slower per job, yet cheaper per unit of work if your workload doesn’t saturate top-tier hardware.

For training, a useful way to think about hardware is:

- **Time to target quality**: how long until the model reaches the accuracy or loss you care about?
- **Cost to target quality**: what did you pay in cloud spend (and possibly energy) to get there?

Suppose you benchmark the same training script on two GPU types:

- GPU A: 1.0x throughput baseline, \$1.00/hour.
- GPU B: 1.8x throughput, \$1.60/hour.

If a run takes 10 hours on A (cost \$10) and 10 / 1.8 ≈ 5.6 hours on B (cost ≈ \$9), GPU B is both *faster* and *cheaper per run*. In that case, the “more expensive” GPU is actually the economical choice. The opposite also happens: if B only gives you 1.1x throughput at 1.6x the price, you’re paying more for less.

Energy has a similar shape. A higher‑power GPU might draw more watts but finish so quickly that the total energy (watt‑hours) per run is comparable or even lower than a slower, low‑power device. What matters is:

- **Energy per completed run** = `average_power_watts * duration_hours`.
- **Energy per unit of work** (for example, per 1,000 inferences or per million tokens).

In practice, several constraints complicate the picture:

- **Queueing and availability**: a slightly worse price/performance device that is always available can beat a “perfect” one you wait hours to schedule.
- **Memory capacity**: devices with more memory allow larger batches or models, which can improve utilization and reduce per-example cost.
- **I/O and networking**: if your workload is I/O-bound, paying for top-tier compute is often wasteful; cheaper hardware with similar I/O characteristics may win.

For inference, latency requirements add another dimension. You might evaluate:

- CPU-only deployment with aggressive batching.
- Small GPU or NPU with moderate batching.
- Larger GPU with many models co-located.

Here, the trade-offs are between:

- **Latency SLOs** (for example, p95 < 100 ms).
- **Hardware utilization** (keeping devices reasonably busy).
- **Cost per 1,000 requests**.

A CPU path might be cheapest at low traffic but become expensive or too slow as QPS grows. A GPU path might look overkill at first but deliver lower cost/1k requests once you cross a certain volume.

The practical workflow is:

1. Shortlist 2–3 realistic hardware options (for example, CPU-only, mid-range GPU, high-end GPU).
2. Run the same small, representative benchmark on each:
   - fixed number of training steps or fixed number of inference requests,
   - measure throughput, cost per run, and approximate energy.
3. Normalize the results into metrics you care about:
   - `time_to_target`, `cost_to_target`, `energy_to_target` for training;
   - `latency`, `throughput`, `cost_per_1k_requests`, `energy_per_1k_requests` for inference.
4. Choose the option that gives you enough performance at the lowest *total* cost (including the engineering effort to support that hardware).

Later in the chapter, we’ll use these ideas when we discuss batching, precision, and scheduling: many “software” tweaks change which hardware tier is attractive by shifting your price/performance curve.

## Software-level levers: batching, precision, and scheduling

Picking efficient hardware is only half the story. The way you *feed* that hardware—how you batch work, which numeric precision you use, and when jobs are allowed to run—often has a bigger impact on cost and energy than switching instance types. These are the knobs you can turn without changing your model architecture.

### Batching: keeping devices busy without blowing up latency

For both training and inference, batching is the main way to turn “many small, inefficient kernel launches” into “fewer, bigger, efficient ones”. Larger batches:

- Improve device utilization and throughput.
- Amortize overheads like Python dispatch, kernel launch, and I/O.
- Can reduce cost per example—up to the point where you run out of memory or hurt convergence/latency.

A minimal training loop with tunable batch size might look like:

```python
# ...existing imports and model setup...
from torch.utils.data import DataLoader

batch_size = 256  # tune this
loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

for step, batch in enumerate(loader, start=1):
    # ...move batch to device, forward, backward, optimizer step...
    pass
```

From an efficiency perspective, you’re looking for:

- GPU utilization climbing toward a healthy range (for example, 60–90%).
- Throughput (examples/sec or tokens/sec) increasing as you grow `batch_size`, *without*:
  - hitting out-of-memory,
  - causing convergence issues,
  - or violating latency SLOs for inference.

On the inference side, batching creates a tension between latency and cost:

- Larger batches → higher throughput and lower cost per request.
- Smaller batches → lower latency for each individual request.

Many serving frameworks (TorchServe, Triton, custom FastAPI+queue setups) support *dynamic batching*: they collect requests for a short window (for example, a few milliseconds) and run them together. You can treat the batching window as an efficiency knob: widen it slightly when you care more about cost, tighten it when you need low latency.

### Precision: trading bits for speed and energy

Most modern accelerators are optimized for reduced precision: FP16, bfloat16, INT8, and friends. Running everything in FP32 is simple, but you may be leaving a lot of performance and energy savings on the table. Reduced precision can:

- Increase arithmetic throughput (more FLOPs per second).
- Reduce memory bandwidth and storage needs.
- Lower energy per operation.

In PyTorch, a typical mixed-precision training setup looks like:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda")
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()

for batch in dataloader:
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad(set_to_none=True)

    with autocast():  # many ops run in FP16/bfloat16 on supported hardware
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

For inference, you can often go further:

- Convert weights to FP16/bfloat16 or INT8 using framework tooling.
- Run the entire forward pass in reduced precision.

The key is to treat precision choices as an *experiment*, not a blind switch:

1. Run a short baseline in FP32: record quality, throughput, and cost per unit of work.
2. Enable mixed precision: check that quality stays within your acceptable range, and record the same metrics.
3. For inference, test quantized variants (for example, INT8) using representative traffic and look for regressions in accuracy and tail latency.

Reduced precision is one of the rare levers that can give you a “win‑win”: higher throughput *and* lower cost/energy per example, with acceptable numerical behavior. But you only know it’s safe if you measure.

### Scheduling: when jobs run and how they share hardware

The third lever is *when* work is allowed to run and how aggressively you pack it onto devices. Two identical training jobs can have very different cost and energy profiles depending on:

- Whether they run on mostly idle hardware vs heavily contended clusters.
- Whether they run during peak cloud pricing or discounted/off‑peak periods.
- How they share GPUs/CPUs with other jobs.

At the simplest level, you can schedule long, flexible jobs during “cheap” times:

- Night hours in your on‑prem cluster (when other teams aren’t using GPUs).
- Cloud provider off‑peak windows or spot instances.

Even with a basic job launcher or cron, you can encode policies like:

```bash
# Run non-urgent training at night
0 1 * * * /usr/bin/python /path/to/train_experiment.py >> /var/log/train_nightly.log 2>&1
```

On more advanced setups (Kubernetes, Slurm, Ray clusters), you can:

- Use lower-priority queues for exploratory or large hyperparameter sweeps.
- Co-locate small, lightweight services to raise overall GPU utilization.
- Enforce per-team or per-project quotas to prevent “noisy neighbors” from wasting shared resources.

From the cost/energy perspective, the questions to ask are:

- Can this job tolerate delay? If yes, place it on cheaper/less contended resources.
- Does this job *need* a whole GPU, or can it share via MIG, MPS, or multiple processes?
- Are there times of day or pricing regimes where the same job is cheaper to run?

You don’t need a full-blown scheduler to benefit. Even simple heuristics—like batching related experiments into a single multi-run script, or deferring non-urgent retrains to low-traffic windows—help you get closer to “maximum useful work per dollar and joule” without touching the model.

In the next sections, we’ll combine these levers into end-to-end training and inference workflows, showing how small changes to batching, precision, and scheduling can significantly shift your cost and energy curves.

## Optimizing training workflows for cost and energy

So far we’ve looked at individual knobs—batch size, precision, hardware choice, scheduling. In practice, you use them together inside a training *workflow*: data loading, forward/backward passes, checkpointing, evaluation, and logging. Small inefficiencies in any of these stages can quietly inflate your cost per experiment.

A cost-aware training workflow usually has three goals:

- Keep expensive devices (GPUs/TPUs) busy with useful math.
- Avoid doing more work than you need to reach your target quality.
- Fail and recover in ways that don’t waste large chunks of progress.

### Structuring the loop: overlap and utilization

A common anti-pattern is a training loop where each step waits on CPU-side work (data loading, preprocessing, logging) before the GPU can do anything. You pay for the GPU the whole time, but it only works in short bursts.

Typical improvements:

- Use `num_workers` and `prefetch_factor` in your dataloader.
- Pin memory and use non-blocking transfers.
- Move cheap preprocessing to the dataloader workers instead of the main loop.

For example, in PyTorch:

```python
# ...dataset and model definitions...
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,      # tune based on CPU cores
    pin_memory=True,    # faster host->device copies
    prefetch_factor=2,  # keep a couple of batches ready
)

device = torch.device("cuda")
model.to(device)

for step, batch in enumerate(loader, start=1):
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    # ...forward, loss, backward, optimizer step...
```

The goal isn’t a particular `num_workers` value; it’s higher measured throughput and utilization with minimal CPU thrash. You can iterate:

1. Start with modest values (e.g., 2–4 workers).
2. Measure `examples/sec` and GPU utilization.
3. Increase workers/batch size until you hit diminishing returns or instability.

### Doing less work: early stopping and smarter evaluation

The cheapest gradient is the one you never compute. If you routinely overtrain models “just in case”, you might be burning a lot of unnecessary GPU hours.

Two simple tactics:

- **Early stopping**: stop when validation metrics stop improving.
- **Progressive evaluation**: eval less frequently early on, more frequently near convergence.

A minimal early stopping sketch:

```python
best_metric = None
patience = 5
bad_epochs = 0

for epoch in range(num_epochs):
    # ...training loop for this epoch...

    val_metric = evaluate(model, val_loader)  # e.g., accuracy, loss

    if best_metric is None or val_metric > best_metric:
        best_metric = val_metric
        bad_epochs = 0
        # ...save best checkpoint...
    else:
        bad_epochs += 1

    if bad_epochs >= patience:
        print("Early stopping triggered.")
        break
```

Even a crude version can save entire epochs of training. The key is to log not just “best accuracy” but also “epochs until stop”, so you can estimate average cost/energy per successful model.

For evaluation frequency, instead of `eval every epoch`, you might:

- Evaluate every *k* epochs early (when you’re far from convergence).
- Reduce `k` as you approach your expected number of epochs.

This cuts down on frequent, expensive full-dataset passes that don’t change decisions.

### Checkpointing and fault tolerance with waste in mind

Long-running jobs fail: preempted instances, cluster maintenance, occasional bugs. If your only checkpoint is at the end of training, a single failure means repeating all the work (and cost).

Efficient checkpointing is a trade-off:

- Checkpoint too often → more I/O, some overhead.
- Checkpoint rarely → more wasted compute on failures.

A pragmatic pattern:

- Save infrequent *full* checkpoints (model + optimizer + scheduler).
- Save more frequent *lightweight* artifacts if needed (for example, EMA weights, key metrics).
- Align checkpoint intervals with real-world events: e.g., every N hours or every M steps.

Sketch:

```python
# ...inside training loop...
if step % checkpoint_interval_steps == 0:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
    }
    torch.save(state, f"ckpt_step_{step:08d}.pt")
```

In cost terms, the question is: *how much work do we expect to lose on average when a job fails?* If the answer is “many hours on expensive hardware”, slightly more frequent checkpoints are usually justified.

### Lightweight tracking of cost and energy per run

You don’t need a full MLOps stack to be cost-aware. Even a small script that estimates cost per run can guide better defaults.

At minimum, record:

- Start/end timestamps.
- Instance type or on-prem cluster label.
- Measured or assumed price per hour.
- Key throughput metrics (from earlier sections).

A minimalist pattern:

```python
import os, time, json

run_meta = {
    "start_time": time.time(),
    "instance_type": os.getenv("INSTANCE_TYPE", "unknown"),
    "price_per_hour_usd": float(os.getenv("PRICE_PER_HOUR_USD", "0.0")),
}

# ...training run...

run_meta["end_time"] = time.time()
duration_hours = (run_meta["end_time"] - run_meta["start_time"]) / 3600.0
run_meta["duration_hours"] = duration_hours
run_meta["estimated_cost_usd"] = duration_hours * run_meta["price_per_hour_usd"]

with open("run_meta.json", "w") as f:
    json.dump(run_meta, f, indent=2)
```

Later, you can parse these small JSON files across runs to answer:

- Which configuration gave us the lowest cost-to-target?
- How much did a particular bug or regression cost us in hardware time?
- Are we trending toward more or less efficient training over time?

The pattern throughout this section is simple: tighten your loop structure, avoid unnecessary work, and make failures cheaper. Combined with the earlier hardware and software levers, this turns “training a big model” from a vague, potentially expensive activity into a repeatable workflow with predictable cost and energy behavior.

## Optimizing inference and serving pipelines

Training might burn a lot of *short-term* compute, but inference and serving often dominate the *lifetime* cost of a model. A slightly inefficient training setup wastes a few GPU-days; an inefficient serving stack can quietly waste GPU-years.

The good news is that most serving inefficiencies come from a handful of design choices:

- How you batch and schedule requests.
- How heavy each model invocation is (precision, architecture, pre/post-processing).
- How tightly you pack workloads onto hardware and scale them up and down.

This section focuses on practical patterns you can implement from Python or your serving framework, without turning your stack into a science project.

### Serving shapes: online, batch, and hybrid

It helps to separate three common shapes:

- **Online (real-time)**: user-facing APIs, tight latency SLOs (for example, p95 < 100 ms).
- **Batch/offline**: daily reports, log processing, large backfills—latency is flexible, throughput and cost dominate.
- **Hybrid**: low-latency path for a subset of traffic, plus asynchronous or cached paths for the rest.

The same model might use different hardware and policies for each shape. For example:

- Online: a smaller, quantized model on a single GPU or CPU autoscaled with traffic.
- Batch: a larger, more accurate model running overnight on cheaper spot instances.

Keeping these shapes separate lets you aggressively optimize batch jobs for cost/energy while keeping the online path simple and predictable.

### Batching and queues: trading latency for efficiency

For online serving, you rarely want to expose batching directly to callers. Instead, use a small **request queue** plus a batching loop inside your service.

A minimal Python sketch (for example, FastAPI + background worker):

```python
import asyncio
import time
from typing import Any, List

import torch
from fastapi import FastAPI

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device).eval()

# Shared queue for incoming requests
request_queue: asyncio.Queue = asyncio.Queue()

class Item:
    def __init__(self, payload: Any):
        self.payload = payload
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()

async def batch_worker(
    max_batch_size: int = 32,
    max_wait_ms: int = 5,
):
    while True:
        # Always wait for at least one item
        first = await request_queue.get()
        batch = [first]
        t_start = time.perf_counter()

        # Collect more items for a short window
        while (
            len(batch) < max_batch_size
            and (time.perf_counter() - t_start) * 1000 < max_wait_ms
        ):
            try:
                batch.append(request_queue.get_nowait())
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
                continue

        # Run a single batched inference
        inputs = collate([b.payload for b in batch]).to(device)
        with torch.no_grad():
            outputs = model(inputs)

        # Split outputs back to futures
        for item, out in zip(batch, split_outputs(outputs)):
            if not item.future.done():
                item.future.set_result(serialize(out))

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())

@app.post("/predict")
async def predict(payload: dict):
    item = Item(payload)
    await request_queue.put(item)
    return await item.future
```

You can tune:

- `max_batch_size`: pushes utilization and throughput up.
- `max_wait_ms`: caps how long a single request can wait in the queue.

From a cost/energy perspective, you’re aiming for:

- Stable latency within your SLO.
- High GPU/CPU utilization during busy periods.
- Lower idle time when traffic dips (scale-to-zero, discussed later).

For batch/offline jobs, the story is simpler: use the *largest batch size* that fits into memory and hits good throughput, then run as many concurrent workers as your hardware and I/O can support.

### Lightening the model: distillation, quantization, and pre/post-processing

Every operation in your serving path costs money and energy, not just the core neural network. You can attack the problem from three angles:

1. **Distillation / smaller architectures**: train a smaller “student” model to mimic a larger “teacher”.
2. **Quantization**: run the model at lower precision (for example, INT8).
3. **Cheaper pre/post-processing**: avoid heavy Python logic, regex storms, or single-threaded feature engineering in the hot path.

Distillation and architecture changes are covered elsewhere in the book; here we’ll look at quantization and pre/post-processing.

A minimal PyTorch static quantization sketch for CPU serving:

```python
import torch
from torch.ao.quantization import (
    quantize_dynamic,
    default_dynamic_qconfig,
)

# Assume `model_fp32` is your trained model
model_fp32 = MyModel()
# ...load weights...

# Dynamic quantization: good for many linear-heavy models on CPU
quantizable_layers = {torch.nn.Linear}
model_int8 = quantize_dynamic(
    model_fp32,
    qconfig_spec=quantizable_layers,
    dtype=torch.qint8,
)

# Save for serving
torch.save(model_int8.state_dict(), "model_int8.pt")
```

You’d then benchmark:

- Latency per request (or per batch) on your target CPU.
- Throughput and cost per 1,000 requests.
- Any accuracy degradation relative to FP32.

For pre/post-processing:

- Move heavy steps into **vectorized** NumPy/PyTorch code where possible.
- Avoid large per-request Python overhead (deep recursion, heavy string processing).
- Cache static or slow-changing features outside the hot path.

Profilers like `cProfile`, PyTorch/TensorFlow profilers, or flamegraphs from your serving framework can quickly show whether you’re spending more time in Python glue than in the model itself.

### Packing workloads onto hardware

For many teams, the biggest serving inefficiency is **underutilized hardware**: GPU instances running at 10–20% utilization because each model gets its own machine.

You can usually do better by:

- Hosting *multiple models* on the same GPU (for example, multi-tenant endpoints).
- Sharing GPUs between training and inference in low-traffic periods (with care).
- Using MIG (Multi-Instance GPU) or similar partitioning features when available.

On a single node, you might run multiple model processes, each pinned to the same GPU but with careful batching and memory budgets. For example, with TorchServe, Triton, or custom process pools, configure:

- Per-model `max_batch_size` and `max_memory`.
- Process counts such that the GPU is busy but not oversubscribed.

On Kubernetes, this often means:

- Larger pods (fewer GPUs per node wasted by sidecars/overhead).
- Tuning `requests`/`limits` so that multiple inference pods can share a node without thrashing.

From Python, you can at least tag and log:

- Which process/node/device handled each request.
- Peak memory usage per model instance.

This gives you the data to decide whether to consolidate or split deployments.

### Scaling policies: matching capacity to demand

Even a perfectly efficient single-node setup wastes energy if you run it at full strength during quiet hours. Autoscaling is the main lever here.

For containerized services (for example, FastAPI + Kubernetes), a simple pattern is:

- Horizontal Pod Autoscaler (HPA) on:
  - CPU/GPU utilization, and/or
  - Requests per second / queue length via custom metrics.
- Scale down aggressively when load drops, allowing idle GPUs to go away or be reassigned.

At a higher level, you can differentiate:

- **Always-on capacity**: minimal number of instances needed to keep cold-start latency acceptable.
- **Burst capacity**: extra instances that can be spot/preemptible or run on cheaper regions.

For batch pipelines, you rarely need long-lived services. Prefer:

- Short-lived jobs that spin up, process a chunk of work, and shut down.
- Scheduling these jobs into low-cost windows (for example, night-time or spot-friendly times).

Even without Kubernetes, you can implement “poor man’s autoscaling”:

- A small controller script that:
  - Checks a queue size (for example, in a database or message broker).
  - Starts or stops worker processes/VMs based on thresholds.

The principle is the same: try not to run expensive hardware when there is no work to do.

### Simple cost and efficiency dashboards for serving

As with training, you don’t need a full observability platform to make better decisions. A minimal logging scheme for inference might record per run or per time window:

- `requests_total`, `requests_success`, `requests_error`.
- `latency_ms_p50`, `latency_ms_p95`.
- `throughput_rps`.
- `avg_gpu_util`, `avg_cpu_util`, `avg_mem_util` where available.
- `cost_usd` for the period (even as a rough estimate).
- `energy_kwh` estimate, if your platform exposes power draw.

Even if you just write a JSON line per 1–5 minutes to a file or object storage, you can later:

- Plot cost per 1,000 requests over time.
- Spot times of day when utilization is low but capacity is high.
- Compare different model versions or deployment configs.

The goal is not perfect accounting; it is to turn “serving feels expensive” into “config X costs ~\$Y/1k requests and uses ~Z Wh/1k requests, and config Y is better/worse by factor K”. Once you have that data, changing batch sizes, quantization levels, or autoscaling thresholds becomes a straightforward engineering exercise rather than guesswork.

## Monitoring, budgeting, and guardrails

Once you start thinking in terms of cost and energy, it’s easy to slip into “we should track everything.” In practice, a small, consistent set of signals is enough to keep your experiments and services within budget and to spot regressions early.

This section focuses on three ideas:

- Collect a *minimal* slice of metrics and metadata for every training run and service.
- Put simple, actionable budgets and alerts around those metrics.
- Add guardrails that keep mistakes from turning into expensive incidents.

### What to monitor: a minimal, practical set

You don’t need a full observability stack to get value. For most teams, the following is enough:

For **training runs**:

- `start_time`, `end_time`, `duration_hours`.
- `instance_type` / node type and region.
- `price_per_hour` (or an approximate on-prem equivalent).
- Core performance metrics:
  - `throughput_examples_per_sec` or `tokens_per_sec`.
  - Final `val_metric` (e.g., accuracy, loss).
- Derived metrics:
  - `estimated_cost_usd = duration_hours * price_per_hour`.
  - Optionally, rough `energy_kwh` per run.

For **serving** (per time window, e.g., per minute):

- `requests_total`, `requests_success`, `requests_error`.
- `latency_ms_p50`, `latency_ms_p95`.
- `throughput_rps`.
- `avg_gpu_util`, `avg_cpu_util` if available.
- `estimated_cost_usd` and `energy_kwh` for the window.

A simple pattern is to emit a compact JSON line for each run or time window and store it in object storage or a log index. For example, at the end of a training run:

```python
import json, os, time

def write_run_summary(path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(meta) + "\n")

# ...run training, collect metrics during/after...
summary = {
    "run_id": os.getenv("RUN_ID", "unknown"),
    "project": "my_model",
    "start_time": start_time,
    "end_time": time.time(),
    "duration_hours": duration_hours,
    "instance_type": instance_type,
    "price_per_hour_usd": price_per_hour,
    "estimated_cost_usd": duration_hours * price_per_hour,
    "throughput_examples_per_sec": throughput,
    "val_metric": best_val_metric,
}
write_run_summary("logs/training_runs.jsonl", summary)
```

Later, you can aggregate these summaries with a small script or notebook instead of needing a full metrics backend.

### Budgets and alerts: turning metrics into action

Monitoring without thresholds still leaves you vulnerable to “surprise” bills. Even lightweight setups can enforce:

- **Per-project or per-team monthly budgets** (for example, \$X/month).
- **Per-experiment cost ceilings** (for example, “any single run above \$Y should be rare and intentional”).
- **Guardrails on training duration** (for example, max wall-clock hours per job).

In managed clouds, you can often:

- Tag resources with `project`, `team`, or `experiment` labels.
- Use native budget tools (e.g., AWS Budgets, GCP Budgets) to:
  - Define cost thresholds per tag.
  - Send email/webhook alerts when thresholds are hit.

In more DIY setups, you can:

- Periodically parse your `training_runs.jsonl` and serving logs.
- Compute spend per project/experiment.
- Emit alerts (email, Slack, etc.) when thresholds are exceeded.

The key is to make alerts **specific and actionable**, for example:

- “Project A spent \$X in the last 7 days (budget \$Y). Top 3 most expensive runs: …”
- “Service B cost per 1,000 requests increased 2× in the last 24 hours; new model version: …”

This keeps budget discussions grounded in concrete runs and configs instead of vague concerns.

### Guardrails: preventing expensive mistakes

Even with budgets, simple mistakes can be costly:

- Launching a huge sweep with a wrong parameter.
- Accidentally training on a high-end instance when a cheaper one is enough.
- Leaving idle resources running over a weekend.

You can reduce the blast radius with lightweight guardrails:

1. **Max resource caps in code**  
   For example, reject obviously unreasonable configurations:

   ```python
   def validate_config(cfg):
       if cfg.num_gpus > 8:
           raise ValueError("num_gpus > 8 requires explicit override.")
       if cfg.max_training_hours > 48:
           raise ValueError("max_training_hours > 48 requires explicit override.")
   ```

   Combine this with a “break-glass” override flag that forces deliberate acknowledgement for expensive runs.

2. **Time-based safeties**  
   Add a simple wall-clock limit for jobs:

   ```python
   import time, sys

   start = time.time()
   max_hours = float(os.getenv("MAX_TRAIN_HOURS", "24"))

   for epoch in range(num_epochs):
       # ...training loop...
       if (time.time() - start) / 3600.0 > max_hours:
           print("Max training hours reached, stopping.")
           sys.exit(0)
   ```

   This won’t catch every issue, but it prevents indefinite runaway jobs.

3. **Idle detection for services**  
   For internal or batch-serving endpoints, track recent traffic:

   - If QPS stays near zero for a long window, scale down or shut down instances.
   - Add “last-used” timestamps to deployments; periodically clean up those unused for N days.

4. **Approval flows for high-cost changes**  
   For infrastructure-as-code or config repos, require review for changes that:

   - Increase instance sizes / counts.
   - Raise per-job limits (max hours, max GPUs).
   - Change autoscaling upper bounds.

   Even a simple code review rule (“two approvals for resource changes”) catches many accidental overshoots.

### Using monitoring data to drive improvements

The final step is to periodically *look* at the data you collect and make simple, concrete decisions:

- Are there experiments with similar quality but very different cost/energy? Prefer the cheaper configs.
- Did a new model version increase cost per 1,000 requests? Decide if the quality gain is worth it.
- Are some services consistently underutilized? Consolidate or scale them down.

Treat these reviews as recurring “efficiency retrospectives”:

- Once per month, plot a few basic graphs:
  - `estimated_cost_usd` per run vs `val_metric`.
  - `cost_per_1k_requests` per model version.
- Capture 1–2 actionable changes (for example, “increase batch size on service X”, “switch project Y to cheaper instance type”).

You don’t need perfect accounting or sophisticated dashboards to get value. The combination of:

- A few metrics per run/service,
- Simple budgets and alerts,
- Basic guardrails in code and configs,

is enough to keep energy and cost efficiency from being an afterthought and turn it into a normal, visible part of how you build and operate AI systems.

## Exercises: estimating and reducing cost per experiment

These exercises are meant to make cost and energy feel as tangible as accuracy and loss. They’re small enough to run on a single machine, but the patterns scale to larger setups.

### 1. Add cost estimation to an existing training script

Take any training script from your own codebase or earlier chapters and:

1. Add environment variables for `INSTANCE_TYPE` and `PRICE_PER_HOUR_USD`.
2. Wrap the training loop to record start/end times.
3. Print and log:
   - `duration_hours`
   - `estimated_cost_usd`
   - `throughput_examples_per_sec` (or tokens/sec)
   - final validation metric

Skeleton:

```python
# ...existing imports...
import os, time, json

run_meta = {
    "start_time": time.time(),
    "instance_type": os.getenv("INSTANCE_TYPE", "local-dev"),
    "price_per_hour_usd": float(os.getenv("PRICE_PER_HOUR_USD", "0.0")),
}

# ...existing training setup and loop...

run_meta["end_time"] = time.time()
duration_hours = (run_meta["end_time"] - run_meta["start_time"]) / 3600.0
run_meta["duration_hours"] = duration_hours
run_meta["estimated_cost_usd"] = duration_hours * run_meta["price_per_hour_usd"]

print(f"Run took {duration_hours:.2f} h, estimated cost ${run_meta['estimated_cost_usd']:.2f}")

with open("run_meta.json", "w") as f:
    json.dump(run_meta, f, indent=2)
```

Questions to answer:

- How much does a “typical” experiment cost on your current hardware?
- Does that align with your intuition?

### 2. Compare two configurations on cost-to-target, not just speed

Pick two training configurations that both reach a target validation metric (for example, 90% accuracy):

- Config A: baseline (current batch size, precision, scheduler).
- Config B: a candidate optimization (larger batch, mixed precision, different hardware type, etc.).

Run each config at least once, collecting:

- `duration_hours`
- `estimated_cost_usd`
- `best_val_metric`
- any relevant throughput metric.

Then compute:

- `cost_to_target` (how much each run cost by the time it hit the metric you care about).
- `speedup` (A vs B in wall-clock time).
- `cost_ratio` (A vs B in dollars).

Prompt yourself:

- Which configuration would you choose if you were paying the bill personally?
- Does that change if you care more about iteration speed than raw dollars?

### 3. Measure the impact of batch size on cost per example

Using a simple model (for example, a small CNN or Transformer on a toy dataset), run short training jobs with different batch sizes, keeping all else fixed:

- `batch_size` in {32, 64, 128, 256, ...} until you hit OOM or instability.

For each run, measure:

- `throughput_examples_per_sec`
- `duration_hours` (or just duration over a fixed number of steps)
- `estimated_cost_usd`
- derived `cost_per_1k_examples`

Plot or tabulate:

- batch size vs throughput
- batch size vs cost per 1,000 examples

Look for:

- Where throughput stops improving significantly.
- Whether cost per 1,000 examples continues to drop, flattens, or starts to rise.

### 4. Simple serving experiment: cost per 1,000 requests

If you have a small inference service (even just a local FastAPI or Flask app), write a short load generator that:

1. Sends a fixed number of requests (for example, 10,000).
2. Measures:
   - total time
   - average and p95 latency
   - requests per second

Run this experiment with two different settings, for example:

- Small batch size or no batching.
- Larger batch size and/or a short batching window.

Estimate:

- `duration_hours` of the run.
- `estimated_cost_usd = duration_hours * price_per_hour`.
- `cost_per_1k_requests`.

Compare:

- How much did cost per 1,000 requests change?
- Did latency stay within acceptable bounds?

### 5. Design a “cost-aware default” for your own project

Based on the earlier sections, sketch (in comments, docs, or a small config file) what a *cost-aware default* would look like for one of your real projects:

- Recommended instance type(s) for:
  - experimentation
  - production training
  - inference
- Default:
  - batch sizes
  - precision settings (FP32, mixed, INT8 for inference)
  - max training hours per run
- Minimal metrics to log for every run or deployment.

The goal is not to be perfect, but to make sure that *the path of least resistance* in your project is also reasonably efficient. Later, you can refine these defaults using real metrics from your own workloads.
