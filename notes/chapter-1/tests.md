# Tests for chapter-1

## Structural
- [ ] S1: The chapter opens with a clear introduction that frames its role as foundational for the rest of the book.
- [ ] S2: The section order in the chapter matches (or very closely follows) the listed sections in the intro outline.
- [ ] S3: Each major section has a level-2 heading (##) with consistent formatting.
- [ ] S4: Code blocks use appropriate language tags (e.g., ```python, ```bash) and render correctly.
- [ ] S5: Learning objectives at the top are all addressed by at least one concrete section or example.

## Clarity
- [ ] C1: The difference between CPUs, GPUs, TPUs, and edge devices is explained in plain language without assuming prior systems knowledge.
- [ ] C2: The role of Python as a “control plane” or orchestrator is described in a way that a reader can restate in their own words.
- [ ] C3: Explanations of libraries and runtimes (NumPy, PyTorch, ONNX, etc.) make their distinct roles clear and non-overlapping.
- [ ] C4: Timing and profiling examples include enough comments or narrative so a reader understands what is being measured.
- [ ] C5: Any caveats about performance measurements (e.g., warmup, data transfer costs) are explicitly stated, not just implied.

## Voice & Tone
- [ ] V1: The tone is practical, calm, and encouraging, avoiding hypey or oversold claims about hardware or performance.
- [ ] V2: The chapter consistently uses second person (“you”) to address the reader and avoids condescension or gatekeeping language.
- [ ] V3: Metaphors and rhetorical questions (e.g., “how do we get the most useful work done…”) support understanding without becoming fluffy or distracting.
- [ ] V4: References to future chapters are confident but not self-congratulatory, and they position this chapter as groundwork rather than a complete guide.
- [ ] V5: The writing maintains a consistent level of technical depth appropriate for an ML practitioner with basic Python experience.

## Reader Impact
- [ ] R1: After reading, a reader should be able to explain why hardware choice affects training speed, inference latency, and cost.
- [ ] R2: A reader should be able to describe, at a high level, how Python delegates heavy computation to lower-level libraries/runtimes.
- [ ] R3: A reader should be able to run at least one timing or profiling snippet from the chapter with minimal environment adjustments.
- [ ] R4: The exercises feel directly connected to the main narrative and clearly reinforce at least one learning objective.
- [ ] R5: The chapter leaves the reader feeling prepared (not overwhelmed) to dive into later, more detailed hardware-specific chapters.

## Chapter-Specific
- [ ] X1: The chapter clearly articulates the idea that modern AI workloads are dominated by a small set of operations (matmuls, convolutions, elementwise kernels) and ties this to hardware design.
- [ ] X2: The explanation of “Python as delegator/control plane” explicitly connects concrete libraries (NumPy, PyTorch, JAX, ONNX, Ray/Dask) to this mental model.
- [ ] X3: The hardware primer section provides at least one concrete, realistic scenario that combines CPUs, GPUs/TPUs, and edge devices in a workflow.
- [ ] X4: The performance trade-offs section explicitly discusses time, money, energy, and engineering complexity as separate axes, not just “fast vs slow.”
- [ ] X5: The quick setup and exercise sections give the reader simple, copy-pastable commands or scripts to verify device visibility and observe a tangible performance difference (e.g., Python vs NumPy, CPU vs GPU).
