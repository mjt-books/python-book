# Tests for chapter-11

## Structural
- [ ] S1: The chapter opens with a clear introduction that explains its focus on TPUs and accelerator-specific patterns and ends with an exercises section.
- [ ] S2: Major sections follow a logical progression from hardware motivation and architecture to JAX/XLA, parallelism, input pipelines, debugging, and exercises.
- [ ] S3: All code blocks are syntactically valid, labeled with appropriate languages, and appear inline with the surrounding explanatory text.
- [ ] S4: Headings and subheadings use a consistent Markdown hierarchy (## for main sections, ### where needed) without skipping levels.
- [ ] S5: The exercises section contains multiple clearly numbered exercises that a reader can follow end-to-end.

## Clarity
- [ ] C1: The chapter clearly explains what makes TPUs different from CPUs/GPUs without assuming prior knowledge of TPU marketing materials.
- [ ] C2: Terms like core, chip, mesh, pod, HBM, JIT, pmap, and pjit are defined or made understandable from context before being used heavily.
- [ ] C3: The text consistently distinguishes between physical meshes and logical meshes in language a non-systems reader can follow.
- [ ] C4: Explanations of compilation vs execution time on TPUs are concrete enough that a reader could time these separately in their own code.
- [ ] C5: Any references to external tools (profilers, dashboards, tf.data, etc.) give just enough description that a reader can recognize them in their environment.

## Voice & Tone
- [ ] V1: The tone is practical and reassuring, emphasizing workflows and mental models instead of low-level hardware minutiae.
- [ ] V2: The prose avoids hypey or marketing-style language about TPUs and instead focuses on trade-offs and realistic use cases.
- [ ] V3: The chapter speaks directly to the reader (“you”) and maintains a consistent instructional voice throughout all sections.
- [ ] V4: Warnings about pitfalls (recompilation, input stalls, sharding mismatches) are framed as guidance, not as blame or gatekeeping.
- [ ] V5: Examples and explanations remain concise and focused, avoiding unnecessary digressions into non-actionable detail.

## Reader Impact
- [ ] R1: A reader who knows basic JAX/NumPy can, after reading, explain when TPUs are worth considering over GPUs for their workload.
- [ ] R2: A reader can outline concrete steps to port an existing JAX training step to TPUs, including handling compilation warmup and device placement.
- [ ] R3: A reader can describe at least one data-parallel and one model-parallel pattern that maps cleanly onto a TPU mesh.
- [ ] R4: A reader comes away with a clear checklist for diagnosing whether a slow job is limited by compilation, compute, communication, or input I/O.
- [ ] R5: The exercises are actionable enough that a motivated reader could run them in a real TPU environment and compare throughput across backends.

## Chapter-Specific
- [ ] X1: The chapter makes explicit the architectural trade-offs of TPUs (specialized matrix engines, static structure, power/cost efficiency) and ties them back to coding patterns.
- [ ] X2: The JAX/XLA section clearly connects purity, shape stability, and tracing behavior to how code actually runs on TPUs, including the “compile once, run many times” model.
- [ ] X3: The parallelism section shows at least one concrete `pmap` example and one `pjit`/logical-mesh example, with clear descriptions of when to use each.
- [ ] X4: The input pipeline section gives at least one realistic prefetching pattern (e.g., tf.data or a custom iterator) and explains how it overlaps host–device transfer with compute.
- [ ] X5: The debugging/profiling section provides a simple timing pattern plus at least one profiling hook example, and explicitly teaches how to interpret “compile vs step” timing on TPUs.
