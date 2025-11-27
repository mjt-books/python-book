# Tools, Utilities, and CI for Training Pipelines (Logging, Metrics, Tracing)

This chapter focuses on the practical tooling that keeps modern AI training pipelines observable, debuggable, and repeatable. You’ll see how to instrument experiments with structured logging and metrics, capture traces for performance debugging, and wire these into continuous integration (CI) so that training code can evolve safely over time.

Instead of treating “observability” as an afterthought, we’ll build it in from the start: every training run should leave behind enough signals to answer questions like “what changed?”, “why did it break?”, and “is it faster or slower than before?”—whether you’re on a laptop, a single GPU server, or a large cluster.

Learning objectives:
- Identify the core observability signals for training pipelines: logs, metrics, traces, and artifacts.
- Compare common Python tools and services for experiment tracking and monitoring.
- Design logging and metrics patterns that scale from local experiments to distributed jobs.
- Understand how to add lightweight CI checks for training scripts and pipelines.

Sections:
- Why observability and CI matter for training pipelines
- Logging fundamentals for ML code (structured logs, contexts, and log levels)
- Metrics and experiment tracking (dashboards, comparisons, and artifacts)
- Tracing and profiling hooks in training workflows
- Integrating tools into local workflows and notebooks
- Adding training-aware checks to CI (smoke tests, regressions, and budgets)
- Exercises: instrumenting a simple training script and adding CI-style checks

## Chapter roadmap

To keep this chapter practical, we’ll follow the lifecycle of a typical training script as it grows from a local experiment into a shared, automated pipeline. Each section introduces one layer of observability or automation and shows how it fits into the broader system.

At a high level, we’ll cover:

1. **Why observability and CI matter for training pipelines**  
   - Ground the chapter with concrete failure stories: silent metric regressions, flaky training runs, mysterious slowdowns.  
   - Define “observability” in the ML context: logs, metrics, traces, artifacts, and configuration.  
   - Connect these ideas to CI: fast feedback on changes before they hit expensive runs or production systems.  
   - Provide a simple mental model: every run should answer “what happened?”, “how well did it work?”, and “how does it compare to before?”.

2. **Logging fundamentals for ML code (structured logs, contexts, and log levels)**  
   - Start with Python’s built-in logging and show how to avoid `print()`-only codebases.  
   - Introduce structured logging (key–value pairs) so logs are machine-readable and filterable across many processes.  
   - Use contextual information (run IDs, ranks, experiment names, hardware info) to make multi-GPU and multi-node logs navigable.  
   - Define log levels and patterns: what belongs at `DEBUG` vs `INFO` vs `WARNING`/`ERROR` in a training pipeline.  
   - Show minimal examples that work both locally and in distributed jobs.

3. **Metrics and experiment tracking (dashboards, comparisons, and artifacts)**  
   - Clarify the difference between logs and metrics, and why you need both.  
   - Introduce simple metric sinks: TensorBoard, CSV/JSON logs, Prometheus-style metrics.  
   - Layer in experiment tracking tools (MLflow, Weights & Biases, custom dashboards) to compare runs, hyperparameters, and hardware configs.  
   - Discuss storing and versioning artifacts: model checkpoints, config files, profiling traces, and sample outputs.  
   - Emphasize stable naming and schemas so metrics remain queryable over months of experiments.

4. **Tracing and profiling hooks in training workflows**  
   - Connect chapter 3’s profiling ideas to day-to-day observability: when to capture traces vs when to just log aggregated metrics.  
   - Show how to integrate lightweight profilers (framework profilers, NVTX ranges, PyTorch/TensorFlow/JAX hooks) into training loops without overwhelming your logs.  
   - Demonstrate targeted profiling: enabling detailed traces only for specific steps, epochs, or CI jobs to catch regressions.  
   - Explain how traces, logs, and metrics complement each other when diagnosing performance issues on GPUs, TPUs, or CPUs.

5. **Integrating tools into local workflows and notebooks**  
   - Start from a realistic baseline: exploratory work in notebooks or small scripts on a laptop or single GPU.  
   - Show low-friction ways to enable logging, metrics, and basic tracking without breaking notebook ergonomics.  
   - Recommend patterns for gradually moving from notebook experiments to “script + config + tracking” while preserving reproducibility.  
   - Discuss how to keep local and remote environments aligned: same logging formats, metric names, and minimal configuration differences.

6. **Adding training-aware checks to CI (smoke tests, regressions, and budgets)**  
   - Define a minimal CI pipeline for ML code: unit tests, style checks, import tests, and fast “smoke” training runs.  
   - Show how to implement tiny end-to-end training checks (do a few steps, verify loss decreases and metrics/logging paths work).  
   - Introduce performance and cost guardrails in CI: catching large slowdowns, GPU overuse, or accidental changes to batch size/sequence length.  
   - Outline patterns for GPU- and accelerator-aware CI: conditional tests, nightly jobs, and staged pipelines (CPU-only checks first, heavy tests later).

7. **Exercises: instrumenting a simple training script and adding CI-style checks**  
   - Provide step-by-step tasks: start from a plain training script, add structured logging, metric emission, and a small set of tracked artifacts.  
   - Extend the script with a “CI mode” that runs only a handful of steps and asserts basic health checks (loss direction, metric shapes, presence of logs).  
   - Encourage readers to wire these pieces into their existing CI system (GitHub Actions, GitLab CI, local runners, or custom schedulers).  
   - Suggest follow-up experiments: adding profiling snapshots to CI, comparing runs across hardware, and evolving a personal or team observability template.

The rest of this chapter assumes you’re comfortable with the basic profiling and hardware concepts from earlier chapters. We’ll focus on turning those one-off checks into durable, observable training pipelines that are easy to debug and safe to change.

## Why observability and CI matter for training pipelines

Most real training pipelines don’t fail with a dramatic crash. They fail quietly: a metric regresses by a few percent, training becomes 30% slower after a harmless-looking refactor, or a rollout ships a model that was trained with the wrong dataset split. By the time someone notices, it’s hard to reconstruct exactly what happened.

Observability and CI are your defenses against this kind of slow, silent drift:

- **Observability** gives you enough signals—logs, metrics, traces, and artifacts—to answer “what changed?” and “what is different between these two runs?”  
- **CI** (continuous integration) makes those questions part of your everyday workflow, not something you only ask after a big failure.

In a non-ML service, CI might just run unit tests and style checks. In an AI training codebase, that’s not enough. A change can:

- Keep all unit tests passing.  
- Still break training by making gradients explode, slowing data loading, or disabling mixed precision.

To catch these issues early, you want each run (including small CI runs) to leave behind a clear trail:

- **Configuration:** which commit, which hyperparameters, which dataset snapshot, which hardware.  
- **Behavior over time:** did loss go down? did throughput and memory usage stay within expected ranges?  
- **Outputs and artifacts:** where are the checkpoints, logs, and small sample predictions stored?

A few concrete scenarios make the value of this obvious:

- You bump a library or CUDA version and suddenly training is 2× slower. With basic observability in place, a quick comparison of step times and GPU utilization between “before” and “after” runs tells you whether the slowdown is real and where it lives (data loading, forward, backward, communication).  
- A colleague changes the preprocessing pipeline, and validation accuracy drops two days later. If your runs capture dataset hashes, feature flags, and evaluation metrics in a consistent way, you can quickly bisect which change caused the regression instead of guessing.  
- You introduce a new parallelism strategy that works on your workstation but fails mysteriously in the CI environment. With structured logs and simple health checks wired into CI, you can see exactly where initialization or communication differs.

The *cost* of adding observability and CI is extra code, more moving parts, and sometimes a bit of performance overhead. The *benefit* is that you stop treating long, expensive runs as black boxes. You can:

- Reproduce interesting behavior (good or bad) because you know which inputs and configs produced it.  
- Compare runs across machines and hardware configurations because metrics and logs share the same schema.  
- Integrate basic “does this still work?” checks into your review process, so regressions are caught before they consume days of GPU time.

In the rest of this chapter, we’ll turn these ideas into concrete patterns:

- First, we’ll standardize how your training code logs what it’s doing.  
- Then we’ll add metrics and experiment tracking so you can see trends and compare runs.  
- We’ll connect profiling and tracing to everyday workflows instead of treating them as special one-off firefighting tools.  
- Finally, we’ll show how to package a small slice of all this into CI jobs, so every pull request gets at least a quick, automated sanity check.

## Logging fundamentals for ML code (structured logs, contexts, and log levels)

Most training scripts start life with a handful of `print()` statements. That’s fine for a single-user experiment, but it breaks down as soon as you:

- Run long jobs in the background or on remote machines.  
- Spawn multiple processes (multi-GPU, distributed data loaders).  
- Need to compare behavior across runs or grep for specific events.

The goal of this section is to move from ad-hoc prints to a *minimal, disciplined logging setup* that still feels lightweight but scales to serious workloads.

We’ll focus on four ideas:

1. Use a logging library instead of raw `print()`.  
2. Prefer **structured logs** over free-form strings.  
3. Attach **context** (run IDs, ranks, hardware) to every log line.  
4. Be intentional about **log levels** so you can dial verbosity up or down.

### 1. Replace `print()` with a real logger

Python’s standard library `logging` module is enough for many projects and plays well with third-party tools.

A minimal pattern for a training script:

```python
import logging
import sys

def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Avoid adding multiple handlers if setup_logging is called more than once
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

logger = setup_logging()

logger.info("Starting training loop")
```

Compared to `print()`, you immediately gain:

- Timestamps.  
- Log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).  
- A central place to control verbosity.

You can keep this function in a small `logging_utils.py` and reuse it across scripts.

### 2. Emit structured logs: key–value, not prose

Plain-text messages are hard for machines to parse at scale. Structured logs keep the human-readable message but add fields that downstream tools can index and filter on.

A simple way to approximate structured logging with the standard library is to embed key–value pairs in a consistent format:

```python
logger.info(
    "train_step end "
    "step=%d loss=%.4f lr=%.1e tokens=%d",
    step,
    loss.item(),
    lr,
    tokens_processed,
)
```

Down the line, you can:

- Grep for `train_step end` to see all step completions.  
- Parse `key=value` pairs to feed dashboards or log search tools.

If you’re ready to depend on a third-party library, packages like `structlog` or JSON-formatting handlers make fully structured logs easy:

```python
# Pseudocode: concept, not a full listing
logger.info(
    "train_step_end",
    step=step,
    loss=float(loss),
    lr=float(lr),
    tokens=int(tokens_processed),
)
```

The important part is the *discipline*: use the same event names and keys across runs so you can compare them later.

### 3. Add context: run IDs, ranks, and hardware info

In a distributed job, “step 100, loss=0.42” is nearly meaningless unless you know:

- Which run it belongs to (commit, config, dataset snapshot).  
- Which process or device emitted it (`rank`, `local_rank`, GPU ID).  
- Potentially, which node or hostname.

You don’t want to repeat this information manually in every log call, so you push it into the logging context.

With the standard library, you can use a `LoggerAdapter` or inject fields into the formatter:

```python
import logging
import sys
import socket
import os

def setup_logging(run_id: str, rank: int = 0) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel("INFO")

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s %(name)s %(levelname)s "
            "run_id=%(run_id)s rank=%(rank)d host=%(host)s %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    # Wrap logger with contextual information
    extra = {
        "run_id": run_id,
        "rank": rank,
        "host": socket.gethostname(),
    }
    return logging.LoggerAdapter(logger, extra)

run_id = os.environ.get("RUN_ID", "local-dev")
rank = int(os.environ.get("RANK", 0))
logger = setup_logging(run_id=run_id, rank=rank)

logger.info("Initialized training")
```

Once this is in place:

- Every log line includes `run_id`, `rank`, and `host`.  
- You can run the same script on a laptop or a cluster and still correlate logs across processes.

In later sections, we’ll reuse the same `run_id` to tie together logs, metrics, and artifacts.

### 4. Use log levels deliberately

Log levels are how you trade off visibility vs noise:

- `DEBUG`: noisy, per-batch or per-step details useful during local debugging.  
- `INFO`: high-level progress and important configuration (epoch boundaries, evaluation metrics, learning rate changes).  
- `WARNING`: something unexpected happened but training can continue (e.g., a batch was skipped).  
- `ERROR` / `CRITICAL`: training is broken or must stop.

A concrete pattern for a training loop:

```python
logger.info(
    "config",
    # if using structured logging, these would be key–value pairs
)
for epoch in range(num_epochs):
    logger.info("epoch_start epoch=%d", epoch)

    for step, batch in enumerate(loader):
        # DEBUG-level: very verbose, disabled in normal runs
        logger.debug("batch_start epoch=%d step=%d size=%d", epoch, step, len(batch))

        # ... forward/backward ...

        if step % log_every == 0:
            logger.info(
                "train_step end "
                "epoch=%d step=%d loss=%.4f throughput=%.1f",
                epoch,
                step,
                loss.item(),
                tokens_per_second,
            )

    logger.info("epoch_end epoch=%d", epoch)
```

A few guidelines that make this workable in practice:

- Keep **configuration dumps** and **environment info** at `INFO` and log them once at startup. They’re invaluable later when you ask “what changed between these two runs?”.  
- Use `DEBUG` only for lines that you’re comfortable turning off in normal runs. Assume `DEBUG` may be disabled in CI or cluster jobs.  
- Don’t log inside tight inner loops at `INFO` unless you batch logs (e.g., every `log_every` steps). Excess logging can become a real performance problem, especially with many processes.

### 5. Making it work in distributed jobs

Multi-GPU and multi-node setups add two extra wrinkles:

- **Multiple processes writing to the same stdout/stderr** can interleave logs.  
- You may only want *one* process (e.g., rank 0) to emit high-level logs to avoid N-way duplication.

Two pragmatic patterns:

1. **Rank-conditional logging**

   Only rank 0 logs at `INFO`, others log at `WARNING` and above:

   ```python
   rank = int(os.environ.get("RANK", 0))
   logger = setup_logging(run_id=run_id, rank=rank)

   if rank != 0:
       logger.setLevel("WARNING")
   ```

   You can still use `DEBUG` logs locally or on a subset of ranks when debugging.

2. **Per-rank log files**

   Instead of writing to stdout, write to a file that includes `run_id` and `rank`:

   ```python
   log_path = f"logs/{run_id}_rank{rank}.log"
   handler = logging.FileHandler(log_path)
   # same formatter as before
   ```

   This makes it easy to inspect a single rank’s behavior (for example, when one worker is slow or crashes) without drowning in interleaved output.

Later, when we talk about metrics, tracing, and CI, we’ll keep building on this foundation: a small, reusable logging setup that:

- Works for single-process experiments and distributed runs.  
- Captures enough structure and context to be machine-readable.  
- Lets you control verbosity without rewriting your training code.

## Metrics and experiment tracking (dashboards, comparisons, and artifacts)

Logs are great for answering “what happened in this run?” but they are less convenient for questions like:

- “Did this change improve validation accuracy over the last three weeks of experiments?”  
- “Which hyperparameters correlate with lower loss on our latest dataset?”  
- “How does throughput differ between the A100 cluster and the local 4090 box?”

For that, you need **metrics and experiment tracking**: structured time series and metadata that can be queried, aggregated, and compared across runs.

In this section we’ll:

1. Clarify what to track (and what not to).  
2. Show simple, file-based metrics logging that works anywhere.  
3. Introduce experiment tracking tools (from “just TensorBoard” up to MLflow/W&B-style services).  
4. Talk about artifacts: where model checkpoints and related files live and how they link back to runs.

### 1. Decide what to measure

Before choosing tools, decide which signals you care about. For most training pipelines, the core buckets are:

- **Optimization metrics**  
  - Training loss, validation loss.  
  - Task-specific metrics (accuracy, F1, BLEU, perplexity, ROC-AUC, etc.).  

- **Performance metrics**  
  - Step time, tokens/images per second.  
  - Data loading time vs compute time.  
  - GPU/CPU utilization, memory usage (even if sampled coarsely).

- **Configuration / metadata**  
  - Commit or version tag.  
  - Hyperparameters (learning rate, batch size, model size, optimizer).  
  - Dataset identifiers (paths, hashes, version IDs, feature flags).  
  - Hardware info (GPU type, number of devices, node name/ID).

You don’t need to log everything at fine granularity. A pragmatic pattern:

- Log **training loss and throughput** every N steps.  
- Log **validation metrics** once per validation run (e.g., every epoch or at fixed wall-clock intervals).  
- Log **config and hardware** once at the start of the run.

Everything should be tied to a **run ID**, the same one you used in logging. That’s the key for joining logs, metrics, and artifacts later.

### 2. File-based metrics: CSV/JSON that work everywhere

The lowest-friction way to get metrics out of your code is to write them to a simple file format (CSV or JSON lines). This works on laptops, clusters, CI, and inside containers without extra infrastructure.

A small CSV logger might look like:

```python
import csv
import os
from pathlib import Path
from typing import Dict, Any

class CSVMetricLogger:
    def __init__(self, run_id: str, out_dir: str = "metrics"):
        self.run_id = run_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{run_id}.csv"
        self._file = self.path.open("w", newline="")
        self._writer = None

    def log(self, step: int, **metrics: Any) -> None:
        row: Dict[str, Any] = {"step": step, **metrics}
        if self._writer is None:
            fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

# Usage in a training loop (rank 0 only)
run_id = os.environ.get("RUN_ID", "local-dev")
metric_logger = CSVMetricLogger(run_id)

for step, batch in enumerate(train_loader):
    # ... forward, backward, optimizer step ...
    if step % log_every == 0:
        metric_logger.log(
            step=step,
            loss=float(loss),
            lr=float(current_lr),
            tokens_per_sec=float(tokens_per_sec),
        )

metric_logger.close()
```

This is deliberately simple:

- Each row is a snapshot: `step, loss, lr, tokens_per_sec, ...`.  
- You can load it later with pandas, plot it in a notebook, or feed it into a dashboard.  
- In distributed jobs, you usually only log metrics from a single process (e.g., rank 0) to avoid duplication.

You can do the same with **JSON lines** (one JSON object per line) if you prefer:

```python
import json

with open(f"metrics/{run_id}.jsonl", "a") as f:
    record = {"step": step, "split": "train", "loss": float(loss)}
    f.write(json.dumps(record) + "\n")
    f.flush()
```

The key discipline is **consistent naming**:

- Use the same metric names (`train_loss`, `val_loss`, `throughput_tokens_per_sec`) across runs and scripts.  
- Use a small set of split identifiers (`train`, `val`, `test`).

This consistency matters more than which file format you pick.

### 3. Dashboards and experiment tracking tools

File-based metrics are enough for an individual developer, but as soon as you have:

- Many runs over time.  
- Several people experimenting simultaneously.  
- Multiple hardware configurations to compare.

you’ll want a dashboard or tracking service. Two common categories:

#### a. Local-first: TensorBoard and friends

TensorBoard is often the first step up from CSVs:

- You log metrics as scalar summaries.  
- It provides web-based plots, run comparisons, and some profiling integration.

A minimal PyTorch-like usage pattern:

```python
from torch.utils.tensorboard import SummaryWriter

run_id = os.environ.get("RUN_ID", "local-dev")
writer = SummaryWriter(log_dir=f"tb_logs/{run_id}")

for step, batch in enumerate(train_loader):
    # ... training ...
    if step % log_every == 0:
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)

# After a validation run
writer.add_scalar("val/loss", val_loss, global_step)
writer.add_scalar("val/accuracy", val_acc, global_step)
writer.close()
```

Advantages:

- Easy to set up, works offline.  
- Runs can be synced or copied to a central place for the team to browse.  

Limitations:

- Comparisons across many projects and hardware configs can get messy unless you standardize directory structures.  
- Auth, sharing, and long-term storage are largely DIY.

#### b. Hosted or self-hosted experiment trackers

Tools like **MLflow**, **Weights & Biases**, **Comet**, and similar systems add:

- A central run registry (with run IDs, configs, metrics, artifacts).  
- Web dashboards for comparing runs by any metadata field.  
- APIs for linking runs to code, issues, and deployments.

A generic pattern (conceptual, not tied to one vendor):

```python
import my_tracking_lib as track

run = track.start_run(
    project="transformer-training",
    run_name="baseline_fp16",
    config={
        "lr": lr,
        "batch_size": batch_size,
        "model": "my_transformer_base",
        "dataset": "dataset_v3",
    },
)

for step, batch in enumerate(train_loader):
    # ... training ...
    if step % log_every == 0:
        track.log_metrics(
            {
                "train/loss": float(loss),
                "train/tokens_per_sec": float(tokens_per_sec),
            },
            step=step,
        )

# After validation
track.log_metrics(
    {
        "val/loss": float(val_loss),
        "val/accuracy": float(val_acc),
    },
    step=global_step,
)

# Save a checkpoint path and other artifacts
track.log_artifact("checkpoints/ckpt_0100.pt", artifact_type="model")
track.end_run()
```

Things to standardize regardless of which tool you pick:

- **Project name**: group related experiments under a consistent project.  
- **Config schema**: store hyperparameters as structured config, not free-form text.  
- **Run naming**: short, descriptive run names help human scanning (`baseline_fp16_seq4k`, `ablate_dropout0.0`, etc.).

When you add CI later, you can:

- Start runs with `run_type="ci_smoke_test"` and much smaller budgets.  
- Use the same tracking tool to see if CI training runs are trending slower or failing more often.

### 4. Artifacts: checkpoints, configs, and traces

Metrics are not the whole story. To make experiments reproducible and debuggable, you need to keep track of the **files associated with a run**:

- Model checkpoints.  
- Optimizer and scheduler state.  
- Training/validation config files.  
- Profiling traces or flamegraphs.  
- Sample outputs (e.g., generated text, predictions, confusion matrices).

Good patterns:

- Use a **directory per run**, named with `run_id`, and put all run-specific artifacts under it:

  ```text
  runs/
    2024-06-01T12-30-00Z_baseline_fp16/
      config.yaml
      metrics.csv
      logs/
        train_rank0.log
      checkpoints/
        epoch_0001.pt
        epoch_0002.pt
      traces/
        step_0100.pytorch_profiler.json
  ```

- Store this directory on a location that’s:
  - Reachable from where you run training.  
  - Backed up or otherwise durable enough for your needs (e.g., object storage in the cloud, a network share on-prem).

- If you use an experiment tracker, register these paths as **artifacts** so the UI links directly to them.

On the code side, it’s often worth introducing a tiny “run context” object early:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunContext:
    run_id: str
    base_dir: Path

    @property
    def run_dir(self) -> Path:
        d = self.base_dir / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def checkpoint_dir(self) -> Path:
        d = self.run_dir / "checkpoints"
        d.mkdir(exist_ok=True)
        return d

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.csv"

# Usage
ctx = RunContext(run_id=run_id, base_dir=Path("runs"))
checkpoint_path = ctx.checkpoint_dir / "epoch_0001.pt"
```

This helps you:

- Avoid scattering path-building logic all over the codebase.  
- Keep runs organized consistently across laptops, servers, and CI.

### 5. Putting it together

By the end of this section, your training loop should:

- Emit structured logs for human-readable troubleshooting.  
- Log a small set of well-named metrics to a file or tracking backend.  
- Save key artifacts under a per-run directory, tied together by a run ID.

In the next section, we’ll layer **tracing and profiling hooks** on top of this foundation, so you can capture detailed performance information for selected runs without overwhelming your everyday metrics and logs.

## Tracing and profiling hooks in training workflows

Metrics tell you *that* a regression happened (“steps are 30% slower”), but they rarely tell you *where* the time or memory went. Profilers and traces fill in that gap: they let you zoom in on a specific run, step, or operation and see a timeline of what the CPUs, GPUs, and framework were actually doing.

The trick is to make profiling a **first-class but selective tool**:

- Available in every training script via a small hook.  
- Cheap enough to leave “off by default”.  
- Easy to enable for a specific run, step range, or CI job when you need detail.

In this section we’ll:

1. Clarify when to reach for tracing vs aggregated metrics.  
2. Show how to wrap a training step in a profiler with minimal code.  
3. Use markers (NVTX or framework scopes) to make traces readable.  
4. Store and link traces alongside other run artifacts.  
5. Sketch how to use “profiling modes” in CI or scheduled jobs.

### 1. When to profile, not just measure

You don’t want full traces for every step of every run; that would be too heavy and noisy. Profiling is most useful when:

- You see a **sudden slowdown** between two commits, environments, or hardware setups.  
- A run is **underutilizing hardware** (e.g., GPUs at 20–30% despite high-level tuning).  
- You suspect a particular phase (data loading, forward, backward, communication) is misbehaving.  
- You need to justify hardware or architecture changes with concrete measurements.

A good rule of thumb:

- Use **metrics** for continuous monitoring and comparisons across runs.  
- Use **profilers/traces** as focused “microscopes” on short windows of time.

That’s why we integrate profiling as a *mode* of the training script: same code, but with a flag or environment variable that turns on detailed tracing for a few dozen steps.

### 2. A minimal profiling wrapper around the training loop

Most major frameworks provide built-in profilers that can capture CPU and GPU timelines and generate artifacts for tools like Chrome’s trace viewer or dedicated UIs.

Here’s a PyTorch-flavored example that keeps the idea general:

```python
import os
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

def should_enable_profile() -> bool:
    # Simple switch: environment variable or CLI flag in real code
    return os.environ.get("PROFILE_TRAINING", "0") == "1"

def run_training(train_loader, model, optimizer, device, run_ctx):
    model.to(device)
    model.train()

    if not should_enable_profile():
        # Normal training loop
        for step, batch in enumerate(train_loader):
            # ... forward, backward, optimizer step ...
            pass
        return

    # Profiling-enabled training loop
    trace_dir = run_ctx.run_dir / "traces"
    trace_dir.mkdir(exist_ok=True)

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=5, warmup=5, active=10, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        with_stack=False,
    )

    with prof:
        for step, batch in enumerate(train_loader):
            # ... forward, backward, optimizer step ...
            loss = model(batch.to(device)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            prof.step()
            if step > 30:  # keep profile window small
                break
```

Key ideas:

- Profiling is gated by `PROFILE_TRAINING` (or a CLI flag). Normal runs pay almost no overhead.  
- We use a **schedule**: a few unprofiled steps, a warmup period, then a short active window.  
- Traces are written under `run_ctx.run_dir / "traces"`, so they tie into the artifact structure from the previous section.

Other frameworks offer similar constructs (e.g., TensorFlow’s `tf.profiler`, JAX’s profiling tools); the pattern—short windows, explicit mode, per-run trace directories—stays the same.

### 3. Add semantic markers to traces (NVTX and scopes)

Raw traces can be hard to read: thousands of kernels and framework internals with unfamiliar names. You can make them much more useful by adding **semantic markers** around important regions, such as:

- “data_loading”  
- “forward_pass”  
- “backward_pass”  
- “optimizer_step”  
- “all_reduce_gradients”

On NVIDIA GPUs, NVTX ranges show up directly in Nsight Systems/Compute and in some framework profilers. Many frameworks also provide their own scoped annotations.

Conceptually, you wrap key regions like this:

```python
import contextlib

@contextlib.contextmanager
def trace_region(name: str):
    # Example with NVTX if available; pseudocode for clarity
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

for step, batch in enumerate(train_loader):
    with trace_region("data_loading"):
        batch = next_batch()  # or just use the for-loop's batch

    with trace_region("forward"):
        out = model(batch.to(device))

    with trace_region("backward"):
        loss = loss_fn(out, targets)
        loss.backward()

    with trace_region("optimizer_step"):
        optimizer.step()
        optimizer.zero_grad()
```

When you open the trace, you see clearly labeled bands for each high-level phase, making it much easier to answer:

- Is data loading actually the bottleneck?  
- How much time is spent in forward vs backward?  
- Where do communication collectives appear relative to compute?

If you already have a structured logging system, you can reuse the same names (`"forward"`, `"backward"`, `"optimizer_step"`) so logs and traces line up conceptually.

### 4. Storing and linking traces with other artifacts

Traces can be large. You don’t want hundreds of them per day, but you *do* want:

- At least one representative trace per major configuration (model size, hardware type, batch/sequence config).  
- Fresh traces when performance changes significantly (after a refactor or dependency bump).

Good practices:

- Put all traces for a run under a predictable directory, e.g.:

  ```text
  runs/
    <run_id>/
      config.yaml
      metrics.csv
      logs/
        train_rank0.log
      traces/
        train_profiler_step000-030/
          local.rank0.trace.json
  ```

- If you use an experiment tracker, log the trace directory as an **artifact** and/or add a URL field:

  ```python
  trace_path = run_ctx.run_dir / "traces"
  track.log_artifact(str(trace_path), artifact_type="trace")
  ```

- In CI or nightly jobs, keep only the most recent N traces per job type to avoid unbounded growth.

Linking traces to the same `run_id` used for logs and metrics means you can:

- Start from “run X was 30% slower than usual” in your dashboard.  
- Jump directly to the corresponding profiler trace to see where the time went.

### 5. Profiling modes in CI and scheduled jobs

Running full profilers in CI for every pull request is usually overkill, but you can still:

- Add **profiling-enabled jobs** to a nightly or weekly schedule.  
- Trigger a profiling run manually for suspicious branches or after big refactors.

A common pattern:

- Define a `--mode` or environment flag with values like:
  - `normal` (default)  
  - `profile_short` (profiling for ~20–50 steps)  
  - `profile_eval` (profiling only evaluation)

- In your CI config:

  - For fast PR jobs: run in `normal` mode with tiny datasets and a handful of steps.  
  - For scheduled jobs: run in `profile_short` mode on a realistic batch/sequence configuration, save traces as artifacts.

Even a single profiling job per day or week can catch:

- Accidental O(N²) operations on growing input sizes.  
- Regressions due to framework or driver upgrades.  
- Changes in communication/computation overlap as you tweak parallelism.

By the end of this section, your training stack should have:

- A clean way to turn profiling on and off without touching core logic.  
- High-level regions annotated so traces are readable.  
- A habit of storing traces next to logs, metrics, and checkpoints for runs that matter.

Next, we’ll look at how to integrate these tools—including logging, metrics, and profiling—into local workflows and notebooks, so you get observability benefits even in the earliest, most experimental stages of a project.

## Integrating tools into local workflows and notebooks

Most real projects start life in a notebook or a small “scratch” script on a laptop or single GPU. That early phase is where ideas are cheap and iteration speed matters more than perfect structure. The risk is that observability and reproducibility become an afterthought—plots live in ephemeral notebook cells, configuration is scattered in global variables, and nothing connects cleanly to the tools you’ll use later on servers or clusters.

This section is about adding *just enough* structure to local workflows that:

- You can still move quickly in notebooks.  
- The same scripts and utilities later plug into CI, clusters, and experiment trackers with minimal changes.

We’ll cover:

1. Light-touch logging and metrics in notebooks.  
2. A simple “run context” that works both locally and remotely.  
3. Patterns for transitioning from notebooks to scripts.  
4. Keeping local and remote environments aligned.

### 1. Light-touch logging and metrics in notebooks

In a notebook, you usually have two kinds of output:

- Immediate feedback in the cell (printed text, plots).  
- Artifacts you might need later (logs, metrics, small artifacts).

You don’t need full-blown logging infrastructure to get started, but you *do* want to avoid having all information trapped in the notebook UI.

A pragmatic pattern:

- Use a minimal logger (from earlier in this chapter) configured to write to both stdout and a file.  
- Use a tiny metrics helper that writes CSV or JSON lines under a per-run directory.  
- Keep the APIs notebook-friendly: short calls that don’t require a lot of boilerplate.

For example, in the first cell of a notebook:

```python
import os
from pathlib import Path
from datetime import datetime

from my_logging_utils import setup_logging          # your helpers from earlier
from my_metrics_utils import CSVMetricLogger        # or inline versions

run_id = os.environ.get("RUN_ID") or datetime.utcnow().strftime("%Y%m%dT%H%M%S_nb")
base_dir = Path("runs")

logger = setup_logging(run_id=run_id, rank=0)
metric_logger = CSVMetricLogger(run_id=run_id, out_dir=base_dir / "metrics")

logger.info("Notebook run started")
logger.info("Hyperparameters lr=%s batch_size=%s", lr, batch_size)
```

Inside training/eval loops, you keep the usage simple:

```python
for step, batch in enumerate(train_loader):
    # ... forward/backward ...
    if step % 50 == 0:
        logger.info("step=%d loss=%.4f", step, loss.item())
        metric_logger.log(step=step, loss=float(loss.item()))
```

You still see log lines in the notebook output, but you also get:

- A `runs/` directory with logs and metrics you can inspect later.  
- A run ID you can reuse if you later move the same experiment into a script.

If you use an experiment tracker (TensorBoard, MLflow, W&B), you can initialize it in a single cell and treat it like a “bonus” sink; the important part is that notebook runs behave similarly to script runs.

### 2. A shared “run context” for notebooks and scripts

Earlier, we introduced a `RunContext` class to organize per-run directories. That same idea works well in notebooks:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunContext:
    run_id: str
    base_dir: Path

    @property
    def run_dir(self) -> Path:
        d = self.base_dir / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def logs_dir(self) -> Path:
        d = self.run_dir / "logs"
        d.mkdir(exist_ok=True)
        return d

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.csv"

    @property
    def checkpoints_dir(self) -> Path:
        d = self.run_dir / "checkpoints"
        d.mkdir(exist_ok=True)
        return d
```

In a notebook:

```python
ctx = RunContext(run_id=run_id, base_dir=Path("runs"))
logger = setup_logging(run_id=ctx.run_id, rank=0)
metric_logger = CSVMetricLogger(run_id=ctx.run_id, out_dir=ctx.run_dir)
```

In a script:

```python
def main():
    run_id = os.environ.get("RUN_ID", make_run_id())
    ctx = RunContext(run_id=run_id, base_dir=Path("runs"))
    logger = setup_logging(run_id=ctx.run_id, rank=rank)
    metric_logger = CSVMetricLogger(run_id=ctx.run_id, out_dir=ctx.run_dir)
    # ... training ...
```

Now:

- Both notebooks and scripts produce artifacts under `runs/<run_id>/...`.  
- CI and cluster jobs can set `RUN_ID` explicitly; local experiments generate one automatically.  
- The rest of your code only needs to know about `ctx`, not about where it’s running.

This is a small abstraction, but it makes it much easier to keep local, remote, and CI runs consistent.

### 3. Transitioning from notebook to script

At some point, a notebook experiment becomes “serious” enough that you want to:

- Reproduce it reliably.  
- Run it on bigger hardware or in CI.  
- Share it with teammates without asking them to execute a specific sequence of cells.

A gentle migration path:

1. **Extract pure-Python functions from notebook cells.**  
   - Move model definitions, dataset loaders, and training loops into a `train_module.py`.  
   - Leave only orchestration (hyperparameter choices, quick plots) in the notebook.  

2. **Introduce a single “entrypoint” function.**  
   - For example, `train(config: TrainConfig, ctx: RunContext)`.  
   - Call this function from:
     - A small `train.py` script that parses CLI flags or reads a config file.  
     - Notebook cells that construct a `TrainConfig` object directly.

3. **Keep configs explicit.**  
   - Avoid relying on global variables or implicit state in the notebook.  
   - Treat the notebook as a UI that builds a config and calls the same underlying function as your scripts.

Conceptually:

```python
# train_module.py
def train(config, ctx):
    logger = setup_logging(run_id=ctx.run_id, rank=0)
    # ... rest of training loop ...

# train.py
if __name__ == "__main__":
    config = parse_args_or_load_config()
    ctx = RunContext(run_id=make_run_id(), base_dir=Path("runs"))
    train(config, ctx)

# notebook.ipynb
from train_module import train, TrainConfig, RunContext

config = TrainConfig(lr=1e-4, batch_size=32, ...)
ctx = RunContext(run_id="debug_nb", base_dir=Path("runs"))
train(config, ctx)
```

This gives you:

- One place to maintain training logic.  
- The ability to run the same code:
  - Interactively in a notebook.  
  - Non-interactively in CI or on a cluster.

### 4. Keeping local and remote environments aligned

Even with shared code, differences between local and remote environments can cause subtle issues:

- Different library versions.  
- Missing system packages.  
- Different CUDA drivers or GPU types.

You won’t eliminate all differences, but you can reduce surprises by:

- **Using the same dependency spec everywhere.**  
  - A `pyproject.toml` or `requirements.txt` that you use:
    - To create a local virtualenv.  
    - As the basis for your Dockerfile or remote environment.  

- **Encapsulating hardware-sensitive settings in config.**  
  - For example:
    - `num_workers` for data loaders.  
    - Batch sizes and sequence lengths.  
  - Let configs differ between “local” and “cluster” profiles, but keep the *structure* the same.

- **Adding a quick “environment dump” to startup logs.**  
  - At the beginning of a run (local or remote), log:
    - Python version.  
    - Key library versions (PyTorch/TF/JAX, CUDA, cuDNN).  
    - GPU model and count if available.  
  - This mirrors the checks in Chapter 1 and makes it much easier to debug “works locally, breaks on the cluster” problems.

For example:

```python
import platform
import torch

logger.info(
    "env python=%s pytorch=%s cuda_available=%s gpu_count=%d",
    platform.python_version(),
    torch.__version__,
    torch.cuda.is_available(),
    torch.cuda.device_count(),
)
if torch.cuda.is_available():
    logger.info("gpu0 name=%s", torch.cuda.get_device_name(0))
```

Because this runs in both notebooks and scripts, you automatically collect comparable environment info across all your runs.

By the end of this section, your local workflows should:

- Use the same logging and metrics patterns as your “serious” training jobs.  
- Store artifacts under the same `runs/<run_id>/...` structure.  
- Call into shared training functions instead of keeping all logic in notebooks.

In the next section, we’ll connect these building blocks to CI: small, training-aware checks that run automatically on each change and prevent silent breakages from sneaking into your main branch.

## Adding training-aware checks to CI (smoke tests, regressions, and budgets)

Traditional CI for Python projects focuses on unit tests, style checks, and maybe a few integration tests. That’s necessary but not sufficient for ML training code. You can pass every unit test and still:

- Break training dynamics (loss no longer decreases).  
- Slow training by 2× through a small change in data loading or model code.  
- Silently stop writing logs, metrics, or checkpoints.

This section shows how to add **training-aware checks** to CI without turning every pull request into a full-scale training run. The goal is to run **small, fast, representative slices** of your pipeline that validate:

- The training loop still runs end-to-end.  
- Key observability hooks (logging, metrics, artifacts) are intact.  
- Basic performance characteristics haven’t regressed catastrophically.

We’ll cover:

1. Designing minimal training smoke tests.  
2. Asserting on “health” signals (loss, metrics, artifacts).  
3. Guarding against performance and cost regressions.  
4. Making CI hardware-aware (CPU-only vs GPU jobs).  
5. Organizing CI configs so they stay maintainable.

### 1. Minimal training smoke tests

A **smoke test** is a tiny training run that answers “does this still basically work?” as cheaply as possible. It should:

- Run in seconds to a couple of minutes.  
- Use a drastically reduced dataset (a few batches or a synthetic dataset).  
- Take just enough optimization steps to see loss move in the right direction.  

You can usually achieve this by giving your training script a `--mode ci` (or similar) flag:

```python
# train.py (conceptual)
def main(args):
    config = load_config(args.config)
    ctx = RunContext(run_id=args.run_id, base_dir=Path("runs"))

    if args.mode == "ci":
        config.num_epochs = 1
        config.max_steps = 20          # cap total steps
        config.dataset_size = 512      # small subset or synthetic data
        config.batch_size = min(config.batch_size, 8)
        config.log_every = 5

    train(config, ctx)

if __name__ == "__main__":
    parser = make_argparser()
    parser.add_argument("--mode", choices=["normal", "ci"], default="normal")
    args = parser.parse_args()
    main(args)
```

In CI, you invoke:

```bash
python train.py --config configs/small.yaml --mode ci
```

The key is that this uses the **same code path** as normal training:

- Same logging setup.  
- Same metrics and artifact handling.  
- Same device selection and distributed initialization (if applicable), just with tiny workloads.

If even this is too heavy for your CI budget, you can move the smoke test into a dedicated test file (e.g., `test_smoke_training.py`) that constructs an even smaller model and dataset but still exercises the end-to-end plumbing.

### 2. Asserting on health signals

A smoke test that merely “does not crash” is helpful but weak. You want it to assert on **simple, robust conditions** that indicate whether training is healthy.

Typical checks include:

- **Loss decreases** over a small window of steps.  
- **Metrics are finite** (no NaNs or infs) and within plausible ranges.  
- **Artifacts exist** where you expect them (logs, metrics files, at least one checkpoint).

For example, a pytest-style smoke test might look like:

```python
# tests/test_smoke_training.py (conceptual)
import math
from pathlib import Path
from my_project.train import train, TrainConfig, RunContext

def test_smoke_training(tmp_path):
    config = TrainConfig(
        # very small model/dataset
        num_layers=2,
        d_model=128,
        batch_size=8,
        max_steps=20,
        log_every=5,
        dataset_size=512,
    )
    ctx = RunContext(run_id="ci-smoke", base_dir=tmp_path)

    history = train(config, ctx)  # return list of loss values or metrics

    # 1) Loss should roughly decrease
    assert len(history) >= 2
    assert history[-1] < history[0] or math.isclose(history[-1], history[0], rel_tol=0.1)

    # 2) No NaNs or infs
    assert all(math.isfinite(v) for v in history)

    # 3) Artifacts exist
    assert (ctx.run_dir / "metrics.csv").exists()
    assert any(ctx.checkpoint_dir.glob("*.pt"))
```

You don’t need perfect statistical guarantees; you just want to catch gross failures:

- Accidentally swapping labels.  
- Removing gradient scaling and causing overflow.  
- Breaking the dataloader so it yields empty batches.

These checks give reviewers confidence that “training still roughly behaves like training” on every change.

### 3. Guardrails for performance and cost

You rarely want to enforce strict performance thresholds in CI (results can be noisy), but simple **guardrails** can catch egregious regressions:

- Maximum allowed wall-clock time for the smoke test.  
- Rough bounds on steps/sec or tokens/sec in a controlled environment.  

Two practical approaches:

1. **Wall-clock timeout in CI**  
   - Rely on your CI system’s job timeout to prevent runaway tests.  
   - Within tests, you can also assert that a small run completes under a loose bound (e.g., `< 120s` on a known runner).

2. **Relative performance checks in scheduled jobs**  
   - For nightly or weekly runs on more stable runners, you can:
     - Log steps/sec to your metrics system.  
     - Compare to a rolling baseline and alert if throughput drops by, say, >30%.  

In code, you might expose a `--max_duration_seconds` flag for CI:

```python
import time

def train(config, ctx, max_duration_seconds=None):
    start = time.time()
    # ... setup ...
    for step, batch in enumerate(train_loader):
        # ... training step ...
        if max_duration_seconds is not None and time.time() - start > max_duration_seconds:
            break
```

Then in CI:

```bash
python train.py --mode ci --max_duration_seconds 90
```

You’re not trying to micromanage every millisecond; you’re just ensuring changes don’t accidentally turn a 30-second smoke test into a 10-minute monster.

### 4. Making CI hardware-aware

Not all CI environments have GPUs (or TPUs). You want your tests to be:

- **Useful without accelerators** (CPU-only).  
- **Able to exploit accelerators when available** (dedicated GPU runners).  

A few patterns help:

- **Device selection in config**, not hard-coded `cuda:0`.  
  - Your training code should accept a `device` or `use_cuda` flag and default to CPU.  

- **Conditional tests based on hardware availability.**  
  - In pytest, you can skip GPU-specific tests if `torch.cuda.is_available()` is false.

  ```python
  import pytest
  import torch

  requires_gpu = pytest.mark.skipif(
      not torch.cuda.is_available(), reason="GPU not available"
  )

  @requires_gpu
  def test_smoke_training_gpu(tmp_path):
      # same as CPU test, but with device="cuda"
      ...
  ```

- **Separate CI jobs for CPU and GPU.**  
  - A common layout:
    - `ci-cpu`: unit tests, style checks, CPU-only smoke tests.  
    - `ci-gpu` (optional): a small GPU-enabled smoke test on a dedicated runner.  
  - If GPU capacity is scarce, you can run `ci-gpu` only on:
    - Main branch.  
    - Nightly schedule.  
    - PRs labeled as “needs-gpu”.

This avoids blocking every contributor on GPU availability, while still giving you periodic signals that GPU training hasn’t silently regressed.

### 5. Keeping CI configs maintainable

CI files (GitHub Actions workflows, GitLab CI configs, etc.) easily become unmanageable if every new check adds copy-pasted blocks. A few guidelines keep things sane:

- **Centralize common commands.**  
  - Use a single script (e.g., `scripts/run_ci_checks.sh`) that:
    - Runs unit tests.  
    - Runs the training smoke test.  
    - Optionally runs linting.  
  - CI jobs then just call that script with environment-specific details.

- **Parameterize jobs with environment variables.**  
  - Example: `CI_MODE=cpu` vs `CI_MODE=gpu` switches:
    - Which test markers to run.  
    - Which devices the training script uses.  

- **Treat CI configs as code.**  
  - Keep them in version control (they usually live in the repo).  
  - Review changes to CI files with the same care as code changes.  
  - Add brief comments explaining why specific timeouts, batch sizes, or datasets were chosen for CI.

Over time, you can evolve your CI from:

- “Just run pytest” →  
- “Run pytest + a small training smoke test” →  
- “Run smoke tests, a GPU check, and occasional profiling or regression jobs on a schedule.”

By the end of this section, your project should have:

- At least one training-aware smoke test that runs in CI.  
- Simple assertions that catch obvious training and observability breakages.  
- A path to layer in performance and cost guardrails without overwhelming your CI budget.

In the final section, we’ll turn everything from this chapter into concrete exercises: you’ll instrument a small training script with logging, metrics, and tracing, then wire a CI-style smoke test around it.

## Exercises: instrumenting a simple training script and adding CI-style checks

These exercises are meant to be small, end-to-end moments: start from a plain training script and gradually add the observability and CI patterns from this chapter. You don’t need a big model or a cluster—most tasks are designed to run on a single CPU or a single modest GPU.

Pick one or two exercises that match your environment and time budget. If you already have a training codebase, feel free to apply the steps there instead of using a toy script.

### 1. Add structured logging to a toy training loop

**Goal:** Replace `print()` calls with a minimal, reusable logging setup.

1. Start from a simple training script (or framework tutorial) that:
   - Trains a small model on a toy dataset (e.g., MNIST, CIFAR-10, or random data).  
   - Prints loss every few steps.

2. Implement a `setup_logging(run_id, rank=0)` helper as described earlier:
   - Logs to stdout with timestamps, levels, `run_id`, and `rank`.  
   - Uses `INFO` for high-level events and `DEBUG` for per-batch detail.

3. Replace raw `print()` calls with:
   - `logger.info("epoch_start epoch=%d", epoch)`  
   - `logger.info("train_step end epoch=%d step=%d loss=%.4f", ...)`

4. Run the script once and inspect the output:
   - Can you grep for `train_step end` and see all steps?  
   - Does each line include the `run_id`?

Optional: Add a `--log-level` flag so you can switch between `INFO` and `DEBUG` without changing code.

### 2. Log metrics to a file and plot them

**Goal:** Emit metrics to a simple file and visualize them after training.

1. Add a `CSVMetricLogger` (or JSON-lines logger) to your script:
   - One row per logging event.  
   - Columns: `step`, `split` (`train`/`val`), `loss`, and at least one additional metric (e.g., accuracy).

2. In your training loop:
   - Log metrics every N steps for training.  
   - Log metrics once per validation run (if you have one).

3. After training, open a notebook or a small script that:
   - Reads the metrics file into pandas (or similar).  
   - Plots `loss` vs `step` for `train` and `val`.

4. Answer for yourself:
   - Does the loss curve match what you saw in the logs?  
   - Is there any obvious bug (e.g., flat or increasing loss) that you would have missed without the plot?

Optional: Add a `run_id` column and run the script twice with different configs, then plot both runs on the same chart.

### 3. Capture a short profiler trace for your training step

**Goal:** Integrate a profiling mode that writes a trace artifact for a short window of steps.

1. Add a flag or environment variable (e.g., `--mode profile` or `PROFILE_TRAINING=1`) to your training script.

2. When profiling is enabled:
   - Wrap ~20–40 training steps in your framework’s profiler (PyTorch, TensorFlow, JAX).  
   - Use a schedule with:
     - A few warmup steps.  
     - A short active window.  
   - Write traces under a `traces/` subdirectory of a per-run directory (e.g., `runs/<run_id>/traces/`).

3. Run the script in profiling mode:
   - Confirm that at least one trace file or directory is created.  
   - Open it in the recommended viewer (TensorBoard, Chrome trace viewer, Nsight, etc.).

4. Try to identify:
   - The relative time spent in data loading vs forward/backward.  
   - Whether GPU utilization looks reasonable during the profiled window.

Optional: Add NVTX or equivalent scopes around `data_loading`, `forward`, `backward`, and `optimizer_step` and verify that they show up in the trace.

### 4. Introduce a “CI mode” and a smoke test

**Goal:** Add a fast-running CI mode to your script and a smoke test that asserts basic training health.

1. Extend your training script with a `--mode` flag:
   - `normal` (default): your usual configuration.  
   - `ci`: override config fields to:
     - Use a much smaller dataset or a synthetic dataset.  
     - Run only a handful of steps (e.g., `max_steps=20`).  
     - Use a smaller batch size.

2. Modify your training function to:
   - Return a small history of recent training losses (e.g., a list of floats).  
   - Write `metrics.csv` and at least one checkpoint into a run directory.

3. Create a test file (e.g., `tests/test_smoke_training.py`) that:
   - Calls your training function with a tiny config and a temporary base directory.  
   - Asserts that:
     - The last loss is lower than (or close to) the first loss.  
     - All logged losses are finite.  
     - `metrics.csv` exists.  
     - At least one checkpoint file exists.

4. Run the test locally (e.g., with pytest) and verify it passes quickly.

Optional: Add a `--max-duration-seconds` parameter and assert that the CI mode run finishes within a loose time bound on your machine.

### 5. Wire your smoke test into your CI configuration

**Goal:** Make the smoke test run automatically on each change.

This exercise is more about CI glue than Python, but it closes the loop.

1. If you use GitHub Actions, GitLab CI, or another service:
   - Add or update a workflow/pipeline file to:
     - Set up your environment (install Python deps, CUDA-only if available).  
     - Run your unit tests (e.g., `pytest`).  
     - Include your smoke test file in the test run.

2. For CPU-only runners:
   - Ensure your training script can fall back to CPU automatically when no GPU is present.  
   - Keep the CI configuration for the smoke test tiny enough to pass on CPU in reasonable time.

3. If you have access to GPU runners:
   - Optionally add a separate job (tagged/labelled appropriately) that runs:
     - The same smoke test, but with `device="cuda"` or equivalent.  
     - Or a slightly larger CI mode that more closely matches real training.

4. Once the pipeline is in place:
   - Make a small, deliberate break (e.g., comment out a log call or change the loss computation) and push a branch.  
   - Confirm that the CI run fails in a way that points you at the right problem (e.g., loss not decreasing, missing metrics file).

Optional: Add a nightly or weekly job that runs your training script in `profile_short` mode and saves traces as build artifacts, giving you a regular performance snapshot.

---

These exercises deliberately reuse the same concepts—`run_id`, structured logs, metrics files, per-run directories, and CI modes—so that by the time you finish, you’ll have:

- A training script that leaves behind a rich, structured record of each run.  
- At least one profiler trace per configuration you care about.  
- Automated checks that catch obvious training and observability regressions before they reach production hardware.

In practice, you can apply the same patterns to much larger codebases: start with a single script or pipeline, instrument it end-to-end, and then gradually extend those conventions across your projects and your team.
