# Tests for chapter-8

## Structural
- [ ] S1: The chapter title clearly states it is about data parallelism, batch sharding, and distributed dataloaders.
- [ ] S2: The chapter includes an explicit learning objectives list near the beginning.
- [ ] S3: All major sections listed in the initial "Sections" list are present and appear in a logical order.
- [ ] S4: Code examples are formatted as fenced code blocks with appropriate language tags (e.g., ```python```).
- [ ] S5: Internal headings form a coherent hierarchy (no jumps in heading levels and consistent use of `##`/`###`).

## Clarity
- [ ] C1: The concept of data parallelism is defined in plain language before deep technical details or APIs are introduced.
- [ ] C2: The distinction between per-device batch size and global batch size is explained with at least one explicit formula or example.
- [ ] C3: The roles of `world_size` and `rank` are described clearly enough that a reader could wire them up in their own script.
- [ ] C4: The responsibilities of samplers/dataloaders vs. the training loop are clearly separated in the explanation.
- [ ] C5: The explanations of communication overhead (compute vs. communication time) avoid unexplained jargon and give at least one intuitive comparison.

## Voice & Tone
- [ ] V1: The tone remains practical and conversational while staying technically precise (no abrupt switches to academic or marketing language).
- [ ] V2: The chapter uses inclusive, second-person language ("you") to guide the reader through the steps.
- [ ] V3: Warnings and "common failure modes" are framed as helpful guidance rather than blaming the reader.
- [ ] V4: When mentioning multiple frameworks (PyTorch, TensorFlow, JAX), the tone stays neutral and avoids strong preference for one.
- [ ] V5: The level of detail stays consistent across sections (no section feels drastically more hand-wavy or overly dense than the others).

## Reader Impact
- [ ] R1: After reading, a practitioner with single-GPU experience should be able to sketch a multi-GPU training loop without external references.
- [ ] R2: The chapter explicitly encourages validating multi-GPU runs against a single-device baseline (e.g., compare loss curves, global batch size).
- [ ] R3: At least one concrete troubleshooting workflow is described for identifying dataloader/sharding bugs.
- [ ] R4: The chapter gives at least one actionable suggestion for improving underutilized GPUs due to input pipeline bottlenecks.
- [ ] R5: The exercises at the end of the chapter are directly implementable and reference concepts or code patterns from the main text.

## Chapter-Specific
- [ ] X1: The chapter explicitly contrasts single-device training with data-parallel training and emphasizes the "replicas + gradient sync" mental model.
- [ ] X2: There is a clear explanation and example of how to use PyTorch `DistributedSampler` (including `set_epoch`) to avoid data duplication and skew.
- [ ] X3: The discussion of distributed dataloaders covers `num_workers`, `pin_memory`, and prefetching, and ties them to GPU utilization.
- [ ] X4: Communication primitives (e.g., all-reduce) are introduced with an intuitive description and connected to how DDP or equivalent abstractions use them.
- [ ] X5: The debugging section includes at least one concrete technique for verifying that all ranks see the correct data shards and synchronized parameters.
