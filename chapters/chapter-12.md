# Orchestration and Scheduling with Kubernetes and Slurm

This chapter introduces how to turn individual training and inference scripts into robust, repeatable jobs that run across clusters. You’ll see how Kubernetes and Slurm express resources (CPUs, GPUs, memory), how they queue and schedule work, and how Python-based AI workloads plug into those systems without becoming deployment horror stories.

Learning objectives:
- Explain the roles of Kubernetes and Slurm in managing AI workloads at scale.
- Map basic hardware needs (GPUs, CPUs, memory, storage) into scheduler/resource specifications.
- Understand core concepts: jobs, pods, nodes, queues/partitions, and priority/preemption.
- Identify when to choose Kubernetes vs Slurm—or how to combine them—in real projects.
- Prepare minimal, reproducible job definitions for common training and inference patterns.

Sections:
- Why orchestration matters once a single machine isn’t enough
- Core concepts: nodes, queues, pods, jobs, and resource requests
- Kubernetes basics for AI workloads
- Slurm basics for AI workloads
- Comparing Kubernetes and Slurm: strengths, weaknesses, and common patterns
- Practical resource specification: GPUs, CPUs, memory, and storage
- Reliability basics: retries, timeouts, and simple health checks
- Exercises: submit, monitor, and tweak a simple GPU job on each system

## Why orchestration matters once a single machine isn’t enough

On a single machine, “scheduling” is usually an afterthought. You run your Python script, maybe pin a few processes to specific GPUs, and call it a day. As long as you remember which terminal is training which model, you can muddle through. That approach collapses the moment you add more users, more GPUs, more nodes—or any pressure to keep expensive hardware busy and shared fairly.

Real clusters have competing demands: researchers launching ad-hoc experiments, production jobs that must finish on time, periodic retraining pipelines, and background preprocessing work. The raw hardware—CPUs, GPUs, memory, storage—is shared, and somebody has to decide who gets what, when. If you handle that informally (“SSH into node-03 and hope nobody else is using it”), you get conflicts, mysterious slowdowns, and idle capacity that still generates a large cloud bill.

Schedulers like Kubernetes and Slurm exist to make these trade-offs explicit. Instead of thinking “run `train.py` on that random GPU box,” you think in terms of *requests*: “run this job with 4 GPUs, 64 GB of RAM, and 2 hours of wall-clock time.” The scheduler decides where to place it, when to start it, and what happens if a node fails or higher-priority work shows up. Your Python code becomes a *job specification* plus a container or environment, not just something you type into a shell.

This shift also changes how you design workflows. Once jobs are first-class objects, you can:

- Run the same training script against different datasets or hyperparameters in parallel, without manually tracking machines.
- Chain jobs into pipelines (preprocess → train → evaluate → deploy) and let the scheduler handle ordering and retries.
- Capture runs as declarative specs (YAML for Kubernetes, submission scripts for Slurm) that can be version-controlled and reproduced later.

From a hardware perspective, orchestration is how you turn a pile of heterogeneous machines into a coherent pool of resources. It’s how you keep GPUs fed without everyone fighting over “the fast node,” and how you move from “heroic one-off experiment” to repeatable training and inference pipelines.

In the rest of this chapter, we’ll treat Kubernetes and Slurm as two different answers to the same core problem: *given a set of jobs and a pool of resources, how do we place, run, and monitor those jobs in a way that is fair, efficient, and debuggable?* Once you can see your Python code through that lens, learning the specific YAML fields or `sbatch` flags becomes much less intimidating—they’re just the syntax for asking the scheduler for what you need.

## Core concepts: nodes, queues, pods, jobs, and resource requests

Before we touch any YAML or job scripts, it helps to normalize the vocabulary. Different systems use different names, but they are mostly describing the same small set of ideas: machines, workloads, and promises about resources.

At a high level you’ll see:

- **Nodes** – the actual machines in your cluster: VMs or bare-metal servers with CPUs, GPUs, memory, and local disks. In Kubernetes, a node runs the kubelet and container runtime; in Slurm, it’s a compute node managed by the controller.
- **Workloads / jobs** – what you want to run: your training script, preprocessing step, or inference service. A “job” is usually one end-to-end invocation of that workload.
- **Queues / partitions** – groupings of nodes plus policies. They encode things like “GPU nodes for research,” “CPU-only nodes for ETL,” or “production jobs only.” Slurm calls these *partitions*; in Kubernetes you approximate them with node labels, taints/tolerations, and namespaces.
- **Resource requests/limits** – what each job *asks for* (and is allowed to use): CPU, memory, GPUs, and sometimes storage or network bandwidth.
- **Priority and preemption** – rules for who goes first when resources are scarce, and which jobs can be paused or evicted to make room.

Kubernetes-specific objects:

- **Pods** – the smallest deployable unit in Kubernetes. A pod usually wraps one or more tightly-coupled containers that share network and storage. For AI workloads, a pod often corresponds to one trainer or one shard of a distributed job.
- **Jobs / CronJobs** – higher-level controllers that create pods and ensure they run to completion (or on a schedule). If a pod fails, the Job object can create a replacement according to its policy.
- **Deployments** – controllers for long-running services (e.g., model inference APIs) that should stay up indefinitely rather than “run once and exit.”

Slurm equivalents:

- **Jobs** – units of work submitted with `sbatch` (batch), `srun` (interactive), or `salloc` (allocation). A job can span multiple nodes, tasks, and GPUs.
- **Partitions** – named groups of nodes with shared limits and policies (max time, max nodes, priority).
- **Job steps / tasks** – sub-units within a job, often mapped to MPI ranks or processes.

From a Python perspective, the most important concept is the **resource request**. Instead of hard-coding “use GPU 0” or “spawn 8 workers,” you describe your needs to the scheduler:

- “I need 4 CPUs, 1 GPU, and 16 GiB of RAM.”
- “I can run on any node with an A100 GPU.”
- “This job should not run longer than 2 hours.”

Kubernetes encodes this in pod specs (`resources.requests` and `resources.limits`); Slurm encodes it in command-line flags or job script directives (e.g., `--gres=gpu:1`, `--cpus-per-task=4`, `--mem=16G`, `--time=02:00:00`).

Two mental shifts make life much easier:

1. **You don’t pick machines; you describe constraints.** The scheduler chooses actual nodes based on availability and policy.
2. **You don’t manage processes directly; you manage jobs.** You submit a specification and ask the system to make reality match it as closely as possible.

The rest of the chapter will keep referring back to these core concepts, just with different syntax in Kubernetes and Slurm. Once you see that a “pod with a GPU request” and a “Slurm job with `--gres=gpu:1`” are answering the same question, cross-learning between the two becomes mostly about translation rather than relearning everything from scratch.

## Kubernetes basics for AI workloads

Kubernetes is a general-purpose orchestration system for containers. For AI work, you can think of it as “a cluster-wide `docker run` with a scheduler attached.” Instead of starting containers by hand on specific machines, you describe *what* should run (images, commands, resources) and let Kubernetes place and manage those containers as **pods** across the cluster.

The usual flow for a Python-based training or inference job looks like this:

1. Package your code and dependencies into a container image.
2. Write a manifest (YAML) describing a pod or Job: which image to use, which command to run, and what resources it needs.
3. Apply that manifest with `kubectl apply -f ...`.
4. Watch Kubernetes create pods, pull images, start containers, restart failed pods if needed, and clean up when they finish.

A minimal mental model:

- **Cluster** – a set of nodes plus a control plane.
- **Node** – a worker machine that can run pods.
- **Pod** – one or more containers that always land together on the same node.
- **Controller** (Job, Deployment, CronJob, etc.) – higher-level objects that manage pods over time.

For batch-style AI jobs (training runs, large preprocessing batches), **Jobs** are the natural fit. A Job says “run N pods until they all succeed (or hit a retry limit).” For online inference, you usually use **Deployments** and **Services** to keep a fixed number of replica pods running behind a stable endpoint.

Here is a stripped-down example of a Kubernetes Job that runs a training script with GPU access:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-model
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: ghcr.io/your-org/your-image:latest
          command: ["python", "-m", "train"]
          resources:
            requests:
              cpu: "4"
              memory: "32Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4"
              memory: "32Gi"
              nvidia.com/gpu: "1"
```

You don’t specify which node to use. Instead, you request resources (`cpu`, `memory`, `nvidia.com/gpu`) and Kubernetes finds a node that can satisfy them. If the cluster is busy, the Job waits in the queue until resources free up.

For AI workloads, a few patterns show up repeatedly:

- **Node selection.** You often want specific GPU types (A100 vs T4) or storage capabilities. Kubernetes lets you express this via node labels and selectors (for example, `nodeSelector: { gpu-type: a100 }`), affinities, or taints/tolerations.
- **Data access.** Training jobs need datasets and checkpoints. You mount these via volumes (network file systems, object storage gateways, or local SSDs) and configure paths in your container.
- **Configuration and secrets.** Hyperparameters, experiment IDs, and credentials live in ConfigMaps and Secrets, mounted as env vars or small files.
- **Observability.** Logs go to `kubectl logs` and often to a centralized logging stack; metrics come from your code and the node’s exporters.

Compared to SSH-based workflows, the main mindset shift is that your “unit of work” is the **manifest**, not the shell command. The YAML file fully describes how to run the job: container image, entrypoint, resources, environment, volumes. Once that is in version control, you can:

- Reproduce past experiments by reapplying old manifests.
- Tweak resource requests or image tags and see how scheduling or performance changes.
- Share runnable examples with teammates without coupling them to a particular machine.

In later sections, we’ll dig into concrete Kubernetes specs for multi-GPU training, distributed jobs, and robust retries. For now, it is enough to see Kubernetes as a declarative way to say: “here is my containerized Python code and the hardware it needs; please run it somewhere in the cluster until it’s done.”

## Slurm basics for AI workloads

If Kubernetes is the dominant container orchestrator, **Slurm** is the dominant scheduler in traditional HPC clusters. Many research labs and supercomputing centers run Slurm, and if you’ve ever been given an account on an “HPC cluster,” there is a good chance `sbatch` is how you’re supposed to run work.

Conceptually, Slurm solves the same problem as Kubernetes: given a pool of nodes and a stream of jobs, decide where and when each job runs. The main differences are:

- Slurm is process- and node-oriented rather than container-oriented.
- It is optimized for batch jobs that may run for hours or days, often using many nodes at once.
- You typically interact with it via command-line tools and small job scripts, not YAML.

A minimal mental model:

- **Controller** – the Slurm scheduler and database, usually running on a “login” or “head” node.
- **Compute nodes** – machines where your jobs actually run.
- **Partitions** – named groupings of nodes with shared policies (e.g., `gpu`, `cpu`, `short`, `long`).
- **Jobs** – units of work you submit with `sbatch` (batch), `srun` (interactive), or `salloc` (interactive allocation).

The usual flow for a Python-based AI job on Slurm is:

1. SSH into a login node.
2. Write a job script with Slurm directives (`#SBATCH ...`) plus your shell commands.
3. Submit it with `sbatch job.sh`.
4. Use `squeue`, `sacct`, and log files to track progress and debug failures.

Here is a stripped-down example of a Slurm batch script for a single-GPU training run:

```bash
#!/bin/bash
#SBATCH --job-name=train-model
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1          # request 1 GPU
#SBATCH --cpus-per-task=4     # 4 CPU cores for data loading, etc.
#SBATCH --mem=32G             # 32 GiB of RAM
#SBATCH --time=02:00:00       # max runtime (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out

# Load modules or activate your environment
module load cuda/12.0
source ~/envs/myproj/bin/activate

# Optional: print some debug info
echo "Running on host $(hostname)"
nvidia-smi

# Run your training script
python -m train
```

You submit this with:

```bash
sbatch train_job.sh
```

Slurm parses the `#SBATCH` lines as **resource requests** and metadata:

- `--partition` chooses which queue/partition to use.
- `--gres=gpu:1` requests a GPU on a node that has GPUs.
- `--cpus-per-task` and `--mem` describe CPU and memory needs.
- `--time` sets a hard wall-clock limit; jobs exceeding it are killed.
- `--output` specifies where stdout/stderr should go, typically including `%j` (job ID).

For AI workloads, some recurring patterns are:

- **Picking the right partition.** Clusters often have several GPU partitions (e.g., `gpu-a100`, `gpu-v100`, `gpu-short`). Each has different limits and policies. Your script usually sets one via `#SBATCH --partition=...`.
- **Balancing CPUs and GPUs.** Data pipelines can bottleneck training if you request too few CPU cores per GPU. `--cpus-per-task` and `--ntasks` give you control over CPU allocation.
- **Interactive runs.** When debugging, you might prefer an interactive shell on a GPU node:
  ```bash
  salloc --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00
  ```
  After this command succeeds, your shell is “inside” an allocation and you can run `python` directly.
- **Array jobs.** For hyperparameter sweeps, Slurm’s job arrays let you launch many similar jobs with a single script and vary parameters by `SLURM_ARRAY_TASK_ID`.

Compared to Kubernetes, the environment and filesystem are usually **shared** on a Slurm cluster: your home directory or project directory is visible from all nodes. That means you often don’t need containers for isolation—you rely on modules, virtualenvs, or Conda environments instead. Some sites *do* combine Slurm with containers (via Singularity/Apptainer or Docker), but the basic interaction model stays the same.

From a Python user’s point of view, your “unit of work” is the **job script**. It fully describes:

- Which resources you need (`#SBATCH` directives).
- How to set up the environment (modules, `conda activate`, etc.).
- The command that actually runs your code.

Once that script is in version control, you can reproduce and tweak experiments just as you would with Kubernetes manifests. Later in the chapter, we’ll look at multi-node, multi-GPU examples and show how to map distributed training patterns onto Slurm’s `--nodes`, `--ntasks`, and `srun` primitives.

## Comparing Kubernetes and Slurm: strengths, weaknesses, and common patterns

By now you’ve seen that Kubernetes and Slurm are two answers to the same question: given jobs and hardware, who runs where and when? In practice, they feel very different to use. It’s useful to have a clear comparison so you can choose the right tool—or understand the constraints of the cluster you’ve inherited.

### High-level contrasts

- **Primary world:**
  - Kubernetes grew up in cloud and microservices land.
  - Slurm grew up in HPC and supercomputing centers.
- **Unit of work:**
  - Kubernetes: pods and controllers (Jobs, Deployments) defined in YAML.
  - Slurm: jobs defined by `#SBATCH`-annotated shell scripts.
- **Packaging:**
  - Kubernetes assumes containers; shipping code usually means building an image.
  - Slurm assumes shared filesystems and environment modules; containers are optional.
- **Workload style:**
  - Kubernetes is great for long-running services, autoscaling, and mixed workloads (APIs, batch, cron).
  - Slurm is great for large, long-running batch jobs and tightly-coupled MPI-style computation.

For Python AI work, both can handle single-GPU jobs, multi-GPU nodes, and multi-node training. The main differences show up in how much ceremony you accept up front (building images, writing manifests) vs how much environment stability you get “for free” (modules, shared `/home`).

### Strengths for AI workloads

Kubernetes tends to shine when:

- You’re deploying **services** (inference APIs, model monitoring, lightweight feature pipelines) that should run 24/7.
- You need to integrate with cloud-native tooling: ingress, autoscalers, secrets managers, service meshes.
- You want **per-job isolation** via containers: consistent dependencies, fewer “it works on my machine” issues.
- Your organization already uses Kubernetes for everything else; adding AI workloads fits existing ops practices.

Slurm tends to shine when:

- You have access to a traditional **HPC cluster** with fast interconnects (InfiniBand), large GPU nodes, and curated modules.
- You run **large, infrequent, heavyweight** training jobs where raw throughput and queue fairness matter more than autoscaling.
- You rely on **shared filesystems** and large datasets mounted centrally (e.g., parallel file systems).
- You want low overhead to get started: write a script, `sbatch`, and you’re in business.

### Weaknesses and rough edges

Kubernetes downsides for AI:

- Requires a **containerization story**: building, pushing, and versioning images adds friction at first.
- Multi-node, multi-GPU training often needs additional components (operators, custom controllers, or launch scripts).
- Storage and networking can be more complex than on a traditional HPC system, especially for large shared datasets.

Slurm downsides for AI:

- **Environment management** can become fragile over time (module versions, per-user Conda envs, subtle differences between nodes).
- It gives you fewer built-in tools for rolling updates, canary deployments, or autoscaling services.
- Multi-tenant isolation is weaker than full container isolation unless the site layers extra tooling on top.

### Common patterns and when to choose which

A few patterns show up repeatedly in real projects:

- **Research cluster with Slurm, production cluster with Kubernetes.**  
  - Train large models on the Slurm cluster where GPUs and high-speed interconnects live.  
  - Export trained models (ONNX, TorchScript, saved weights) and deploy them behind HTTP/gRPC services in Kubernetes.
- **Single orchestration system, both training and serving.**  
  - Kubernetes: use Jobs for training, Deployments for serving.  
  - Slurm: use batch jobs for both training and offline inference; serve only minimal online components elsewhere.
- **Hybrid “containers on Slurm”.**  
  - Package environments as containers (e.g., Apptainer/Singularity images) but schedule them with Slurm, combining reproducibility with HPC-friendly scheduling.

As a rough decision heuristic:

- If you mostly run **batch experiments** on a shared research cluster, and IT hands you a login to a Slurm system, lean into Slurm and make your job scripts solid.
- If you mostly build **services and pipelines** in a cloud environment, and your org already runs Kubernetes, treat Kubernetes as your default and add AI workloads to that ecosystem.
- If you can choose freely, ask where your **ops and platform teams** have the most experience. A moderately tuned system that fits existing workflows usually beats a theoretically perfect system that nobody knows how to run.

Throughout the rest of this chapter, we’ll use this comparison as a translation guide: for each pattern (resource requests, retries, multi-node jobs), we’ll show how it looks in both Kubernetes YAML and Slurm job scripts so you can move between them when your hardware or organization demands it.

## Practical resource specification: GPUs, CPUs, memory, and storage

Conceptually, “resource requests” are simple: tell the scheduler what you need. In practice, small changes here can double queue times or cut training speed in half. This section focuses on *good-enough defaults* and how to express them in both Kubernetes and Slurm.

### Start from the workload, not the cluster brochure

Rather than starting with “this node has 8 GPUs,” start with questions like:

- How many GPUs can this training script actually keep busy?
- How many CPU cores do I need to feed each GPU (data loading, preprocessing)?
- How much RAM and GPU memory does a single process use at peak?
- Do I need fast local scratch space, or is network storage fine?

Run a small profiling pass on a single machine (as in chapter 1). Use that to set initial per-job requests, then scale out.

### Rule-of-thumb baselines

You can adjust these later, but they’re reasonable starting points:

- **GPUs**
  - 1 GPU per trainer process.
  - If you’re not using distributed training yet, start with `1` and scale up only when you have a clear plan.
- **CPUs**
  - 2–4 CPU cores per GPU for typical PyTorch/TensorFlow jobs with moderate data loading.
  - Increase if you do heavy preprocessing or complex augmentations on CPU.
- **Memory**
  - System RAM: 4–8 GiB per GPU as a starting point for training jobs.
  - GPU memory: driven by model size and batch size—measure with `nvidia-smi` or framework tooling.
- **Storage**
  - Local scratch (SSD) for temporary files and shuffled datasets if available.
  - Network storage for shared datasets and checkpoints.

### Expressing this in Kubernetes

A single-GPU Job with 4 CPUs, 32 GiB RAM, and a labeled GPU node might look like:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-model
spec:
  template:
    spec:
      nodeSelector:
        gpu-type: a100               # only run on A100 nodes
      restartPolicy: Never
      containers:
        - name: trainer
          image: ghcr.io/your-org/your-image:latest
          command: ["python", "-m", "train"]
          resources:
            requests:
              cpu: "4"
              memory: "32Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4"
              memory: "32Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: dataset
              mountPath: /data
      volumes:
        - name: dataset
          persistentVolumeClaim:
            claimName: imagenet-pvc
```

Notes:

- Use **requests** to tell the scheduler what you need; **limits** to cap usage. For dedicated GPU jobs, keeping them equal avoids throttling surprises.
- Use **labels + nodeSelector/affinity** to steer jobs to the right hardware instead of hard-coding node names.
- Attach storage via **volumes**; avoid baking large datasets into the image.

### Expressing this in Slurm

The same intent in Slurm is mostly `#SBATCH` flags:

```bash
#!/bin/bash
#SBATCH --job-name=train-model
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GiB RAM
#SBATCH --time=04:00:00           # generous upper bound
#SBATCH --output=logs/%x-%j.out

# Optional: constrain to nodes with local SSDs or other features
# #SBATCH --constraint=ssd

# ...environment setup...

python -m train \
  --data-root /datasets/imagenet \
  --output-dir /scratch/$USER/experiments/run-$SLURM_JOB_ID
```

Notes:

- **Partition** selection approximates Kubernetes’ node labels: different GPU types, short/long queues, etc.
- Use `--constraint` or site-specific features if you need specific node capabilities.
- Put large temporary files and checkpoints on node-local or high-performance storage when available (often `/scratch`, `/tmp`, or a site-specific path).

### Iterate with measurement

Once the job runs, measure:

- **Utilization** (`nvidia-smi`, framework profilers) to see if GPUs are starved by insufficient CPUs or I/O.
- **Peak memory** (system and GPU) to avoid over-requesting “just in case.”
- **Queue times**: if jobs sit pending due to over-ambitious requests (e.g., 8 GPUs when 1 would do), split them into smaller, more schedulable units.

The goal is not a perfectly optimized spec on day one, but a **reasonable starting point** that you refine based on real runs. The next section will build on this by adding reliability tools—retries, timeouts, and simple health checks—so your well-sized jobs also behave well when something goes wrong.

## Reliability basics: retries, timeouts, and simple health checks

On a laptop, when a training script crashes you just re-run it. On a cluster, that approach does not scale: jobs fail due to preemptions, node issues, transient storage problems, and the occasional bug. Schedulers give you basic knobs to absorb these bumps so you don’t have to babysit every experiment.

We’ll focus on three simple tools:

- **Retries** – automatically re-run failed jobs a small number of times.
- **Timeouts** – enforce upper bounds on runtime so stuck jobs don’t burn days of GPU time.
- **Health checks** – cheap signals that a job is still making progress.

### Retries

Retries protect you from transient issues: node reboots, flaky network storage, temporary quota errors. You *don’t* want infinite retries—if a bug in your code segfaults instantly, it will just fail faster forever—so keep the retry count modest and make logs easy to inspect.

**Kubernetes Jobs** have built-in retry support via `backoffLimit` and `restartPolicy`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-model
spec:
  backoffLimit: 3          # fail job after 3 failed pod attempts
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: trainer
          # ...existing container spec...
```

Typical pattern:

- `restartPolicy: OnFailure` – if the container exits with a non-zero code, create a new pod.
- `backoffLimit: 3–5` – enough to ride out flaky nodes, not enough to hide persistent bugs.

In **Slurm**, retries are up to you. The simplest pattern is a thin wrapper script:

```bash
#!/bin/bash
# retry_wrapper.sh
set -euo pipefail

MAX_RETRIES=3
RETRY_DELAY=60   # seconds

for attempt in $(seq 1 "$MAX_RETRIES"); do
  echo "Attempt $attempt/$MAX_RETRIES"
  if python -m train; then
    echo "Success on attempt $attempt"
    exit 0
  fi
  echo "Attempt $attempt failed"
  if [[ "$attempt" -lt "$MAX_RETRIES" ]]; then
    echo "Sleeping $RETRY_DELAY seconds before retry"
    sleep "$RETRY_DELAY"
  fi
done

echo "All attempts failed"
exit 1
```

Then your Slurm script runs `python -m retry_wrapper` (or just calls this shell script). Slurm sees the overall job as a single unit; the retries happen inside it.

### Timeouts

Timeouts cap how long a job can run. This protects the cluster from runaway jobs (infinite loops, deadlocks) and helps you catch mis-specified workloads.

In **Kubernetes**, you typically combine:

- Job-level `activeDeadlineSeconds` – kill pods if they run longer than this many seconds.
- Pod/container-level timeouts in your own code (for HTTP calls, data loads, etc.).

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-model
spec:
  activeDeadlineSeconds: 14400   # 4 hours
  backoffLimit: 3
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: trainer
          # ...existing container spec...
```

In **Slurm**, `--time` is mandatory on many clusters and acts as a hard wall-clock timeout:

```bash
#SBATCH --time=04:00:00   # HH:MM:SS
```

If your job hits this limit, Slurm sends signals and then kills it. Use a value that is:

- Slightly above your expected runtime, based on smaller test runs.
- Not so large that genuine hangs take days to notice.

### Simple health checks

For batch jobs, “health” mostly means “is it still making progress?” You don’t usually add full HTTP health endpoints, but you *do* want signals that differentiate “working slowly” from “stuck forever.”

Useful low-tech patterns that work in both systems:

- **Heartbeat logs** – periodic log lines like `step=1000, loss=...` so you can see fresh output via `kubectl logs` or Slurm log files.
- **Progress files** – write a small JSON or text file with the latest completed epoch/step; useful for monitoring dashboards or debugging restarts.
- **Exit codes** – ensure your script exits non-zero on serious failures so retries and alerts can trigger.

Kubernetes gives you more formal probes for long-running services (Deployments, not Jobs):

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

For training Jobs, inline heartbeats plus Job status are usually enough.

On Slurm, a common pattern is to log timestamps around major phases:

```bash
echo "$(date --iso-8601=seconds) - starting training"
python -m train
status=$?
echo "$(date --iso-8601=seconds) - training finished with status $status"
exit "$status"
```

This makes it obvious from the log whether a job never started, got stuck mid-run, or finished cleanly.

### Putting it together

The goal is not a perfect, self-healing system, but a **minimal reliability baseline**:

- Every job has:
  - A sensible **time limit**.
  - A small **retry budget** for transient issues.
  - Logs that show regular **progress** and a clear exit status.

Once you have that in place, you can safely scale up: larger jobs, more nodes, and more complex pipelines. The final section of this chapter will turn these ideas into concrete exercises: submitting, monitoring, and iterating on a simple GPU job in both Kubernetes and Slurm.

## Exercises: submit, monitor, and tweak a simple GPU job on each system

These exercises are designed to be small, self-contained steps that turn the concepts in this chapter into something you can actually run. You don’t need a giant model or a huge cluster—just a single GPU node accessible via Kubernetes or Slurm (or both).

Adapt the commands to your environment; the important part is the *shape* of the workflow: define a job, submit it, watch it run, and then tweak resource and reliability settings.

### 1. Run a simple GPU Job on Kubernetes

1. Create a minimal training-like script inside your container image (or reuse an existing one). It should:
   - Detect the GPU (e.g., print `torch.cuda.is_available()` and device name).
   - Run a tiny loop (a few matmuls or training steps) and log progress.
2. Write a Kubernetes Job manifest `gpu-job.yaml` similar to the example in this chapter:
   - Request 1 GPU, 4 CPUs, and ~16–32 GiB of RAM.
   - Set `restartPolicy: OnFailure` and `backoffLimit: 3`.
3. Apply the manifest and observe behavior:
   - `kubectl apply -f gpu-job.yaml`
   - `kubectl get jobs,pods`
   - `kubectl logs <pod-name>`
4. Modify the manifest:
   - Increase CPU requests and re-run; watch if GPU utilization improves.
   - Add `activeDeadlineSeconds` with an unrealistically small value and see how the Job fails.

Write down:

- How long the job takes.
- What you see in the Job and Pod status when it succeeds vs when it times out or fails.

### 2. Submit the same workload via Slurm

1. Put the same Python script on a Slurm-accessible filesystem (e.g., your home directory).
2. Write a Slurm batch script `train_job.sh`:
   - Request 1 GPU, 4 CPUs, and ~16–32 GiB RAM.
   - Set a time limit (`--time=00:10:00`) and an output log path.
3. Submit and monitor:
   - `sbatch train_job.sh`
   - `squeue -u $USER`
   - Inspect the log file in `logs/`.
4. Add a retry wrapper as shown earlier, and modify the Python script to occasionally fail (for example, raise an exception with small probability). Verify that:
   - The wrapper retries a few times.
   - The final Slurm job exit code reflects overall success/failure.

Write down:

- How long the job waits in the queue.
- How failures and retries show up in Slurm logs vs Kubernetes logs.

### 3. Compare resource specifications and their effects

Pick one knob to tune and compare its impact on both systems:

- Option A: **CPU per GPU**
  - Run with 1 CPU per GPU, then 4 CPUs per GPU.
  - Measure GPU utilization and total runtime.
- Option B: **Time limits**
  - Set a time limit that is slightly below the real runtime and observe how each system kills the job and reports the error.
- Option C: **Memory**
  - Intentionally under-request memory (within safe limits) and see how OOM conditions are reported (container termination reason in Kubernetes vs Slurm OOM messages).

For each run:

- Capture the exact Job/Slurm configuration (YAML or `#SBATCH` lines).
- Capture the command you used to inspect status (`kubectl get`, `squeue`, `sacct`, etc.).
- Note how easy it is to diagnose what went wrong from logs and status alone.

After you finish, you should have:

- At least one working GPU job definition for Kubernetes and/or Slurm that you can reuse later in the book.
- A concrete feel for how small changes to resource requests and reliability settings affect queueing, runtime, and failure behavior.
- A starting point for more advanced patterns in later chapters (multi-node jobs, distributed training, and automated pipelines).
