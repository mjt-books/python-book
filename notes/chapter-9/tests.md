# Tests for chapter-9

## Structural
- [ ] S1: The chapter opens with a clear explanation of what model parallelism is and why it matters in the broader context of scaling deep learning models.
- [ ] S2: Major sections follow a logical progression from motivation, to forms of model parallelism, to memory/communication details, to frameworks, debugging, and exercises.
- [ ] S3: Each section has descriptive headings and subsections are scoped so that no single section feels overloaded or unfocused.
- [ ] S4: The exercises section is clearly separated and uses a numbered list with actionable tasks.
- [ ] S5: Cross-references to other chapters or concepts (e.g., data parallelism, profiling) are consistent with the book’s overall structure and terminology.

## Clarity
- [ ] C1: The chapter explicitly contrasts model parallelism with data parallelism in simple terms before diving into details.
- [ ] C2: Each form of model parallelism (layer, tensor, pipeline, experts) is defined with at least one concrete, easy-to-visualize example.
- [ ] C3: Technical terms like all-reduce, all-gather, reduce-scatter, and process groups are briefly explained or put in intuitive language when first introduced.
- [ ] C4: The explanations of memory components (parameters, activations, optimizer state, checkpoints) make it clear which ones live on-device vs off-device.
- [ ] C5: Any discussion of communication pitfalls avoids ambiguous phrases and makes it clear what goes wrong and how to notice it.

## Voice & Tone
- [ ] V1: The tone remains pragmatic and coaching-oriented, emphasizing trade-offs and when to reach for model parallelism rather than hyping it.
- [ ] V2: The chapter consistently uses inclusive “you” and “we” language without drifting into overly academic or overly informal phrasing.
- [ ] V3: Warnings about complexity, debugging difficulty, and performance pitfalls are framed constructively, with suggestions for how to cope.
- [ ] V4: Examples and recommendations are neutral with respect to specific vendors or hardware, matching the rest of the book’s style.
- [ ] V5: The chapter avoids unexplained jargon and assumes a competent but non-expert reader in distributed systems.

## Reader Impact
- [ ] R1: By the end of the chapter, a reader should be able to decide when data parallelism is sufficient and when model parallelism is warranted for their workload.
- [ ] R2: The reader is given a clear mental model for how memory is distributed across devices under different model parallel strategies.
- [ ] R3: The chapter provides at least a rough checklist or workflow for profiling and improving a model-parallel training run.
- [ ] R4: The exercises are realistically completable by a motivated practitioner with access to a few GPUs, without requiring giant clusters.
- [ ] R5: The reader comes away with a sense of which libraries to investigate next (PyTorch, DeepSpeed, Megatron, etc.) and what each is roughly good for.

## Chapter-Specific
- [ ] X1: The “When and why model parallelism is necessary” section clearly states that model parallelism is a tool to use after exhausting simpler options like batching and data parallelism.
- [ ] X2: The “Forms of model parallelism” section distinctly differentiates layer-wise, tensor, pipeline, and expert parallelism, including their main trade-offs and typical use cases.
- [ ] X3: The “Memory anatomy” section explicitly ties parameter, activation, and optimizer state placement to how sharding and model parallel choices reduce per-device memory.
- [ ] X4: The “Communication patterns and performance pitfalls” section includes at least one concrete list of common pitfalls (e.g., too many small messages, imbalanced stages, blocking communication) and links them to observable symptoms (idle GPUs, hangs, etc.).
- [ ] X5: The “Framework support” and “Debugging and monitoring” sections give the reader at least one actionable starting point per framework/tool (e.g., which API to look at, which profiler/trace to enable) rather than only naming libraries.
