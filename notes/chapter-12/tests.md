# Tests for chapter-12

## Structural
- [ ] S1: The chapter opens with a clear introduction that frames orchestration/scheduling problems and ends with an exercises section focused on running real jobs.
- [ ] S2: Major sections follow a logical progression from why orchestration matters, through core concepts, Kubernetes and Slurm basics, comparison, resources, reliability, and exercises.
- [ ] S3: All code blocks (YAML, Bash, etc.) are syntactically plausible, labeled with appropriate languages, and appear close to the explanatory text that motivates them.
- [ ] S4: Headings and subheadings use a consistent Markdown hierarchy (## for main sections, ### where needed) without skipping levels or mixing styles.
- [ ] S5: The exercises section contains multiple clearly numbered exercises that a reader can follow end-to-end on Kubernetes and/or Slurm.

## Clarity
- [ ] C1: The chapter clearly explains why manual SSH-based workflows break down at cluster scale and how schedulers address those issues.
- [ ] C2: Terms like nodes, queues/partitions, pods, jobs, resource requests/limits, priority, and preemption are defined or made understandable from context before being used heavily.
- [ ] C3: The text explicitly maps equivalent concepts between Kubernetes and Slurm (e.g., partitions vs node selectors, Jobs vs `sbatch`) so readers can translate between them.
- [ ] C4: Examples of resource specification (GPUs, CPUs, memory, storage) are concrete enough that a reader could adapt them to their own cluster configuration.
- [ ] C5: Explanations of retries, timeouts, and health checks are specific about which fields/flags to use in each system and how failures will appear in logs/status.

## Voice & Tone
- [ ] V1: The tone is practical and non-dogmatic, presenting Kubernetes and Slurm as tools with trade-offs rather than promoting one as universally superior.
- [ ] V2: The prose avoids vendor-specific hype and instead focuses on realistic cluster behaviors, constraints, and failure modes.
- [ ] V3: The chapter speaks directly to the reader (“you”) and maintains a consistent instructional voice across all sections.
- [ ] V4: Descriptions of operational pitfalls (queue delays, mis-specified resources, stuck jobs) are framed as guidance and checklists, not as blame or gatekeeping.
- [ ] V5: Examples and explanations remain concise and actionable, avoiding deep digressions into scheduler internals that are not needed for typical AI workloads.

## Reader Impact
- [ ] R1: A reader can explain the roles Kubernetes and Slurm play in managing AI workloads and why orchestration becomes necessary beyond a single machine.
- [ ] R2: A reader can map basic hardware needs (GPUs, CPUs, memory, storage) into concrete Kubernetes resource specs and Slurm `#SBATCH` directives.
- [ ] R3: A reader can describe when Kubernetes vs Slurm is a better fit for a given project scenario and recognize common hybrid patterns (e.g., train on Slurm, serve on Kubernetes).
- [ ] R4: A reader can write at least one minimal, reproducible job definition for each system (a Kubernetes Job manifest and a Slurm batch script) for a simple GPU training run.
- [ ] R5: After completing the exercises, a reader can submit, monitor, and tweak a simple GPU job on their available scheduler and interpret queue states and logs.

## Chapter-Specific
- [ ] X1: The “Core concepts” section clearly normalizes vocabulary across systems (nodes, jobs/workloads, queues/partitions, resource requests, priority/preemption) and emphasizes describing constraints rather than picking machines.
- [ ] X2: The Kubernetes section provides a concrete Job manifest for an AI workload that includes image, command, GPU/CPU/memory requests, and optionally node selection and volumes.
- [ ] X3: The Slurm section provides a concrete batch script for a GPU training job that includes partition, GPU request (`--gres`), CPU/memory, time limit, and log configuration.
- [ ] X4: The resource specification section gives rule-of-thumb baselines (e.g., CPUs per GPU, RAM per GPU) and shows how to encode them in both Kubernetes YAML and Slurm directives.
- [ ] X5: The reliability section demonstrates retries, timeouts, and simple health/progress signals in both systems, including specific fields/flags (`backoffLimit`, `activeDeadlineSeconds`, `--time`, wrappers) that a reader could copy.
