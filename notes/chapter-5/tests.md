# Tests for chapter-5

## Structural
- [ ] S1: The chapter starts with a clear introduction that explains its purpose in the context of the overall book.
- [ ] S2: The major sections follow the order outlined in the "Sections" list near the top of the chapter.
- [ ] S3: Code examples are formatted as fenced code blocks with the correct language tags (e.g., `python`, `cpp`).
- [ ] S4: Each section has a descriptive heading and there are no large, unstructured walls of text.
- [ ] S5: The Exercises section contains clearly numbered or titled exercises that map to the chapter’s main ideas.

## Clarity
- [ ] C1: The differences between CPUs and GPUs are explained in plain language before introducing technical terms like SIMT, warps, and grids.
- [ ] C2: The explanations of host vs device memory, transfers, and pinned memory can be understood without prior CUDA experience.
- [ ] C3: Kernel-related concepts (threads, blocks, grids, kernels, launch overhead) are described at an intuitive level, without requiring C++ knowledge.
- [ ] C4: Each code sample includes enough context that a reader could run it with minimal assumptions (e.g., imports, device checks).
- [ ] C5: Any new jargon (e.g., “coalesced access”, “divergence”, “pinned memory”) is either briefly defined or made clear from context.

## Voice & Tone
- [ ] V1: The tone stays friendly and reassuring, emphasizing intuition over low-level CUDA expertise.
- [ ] V2: The text consistently addresses the reader as a Python user and avoids assuming expert systems or GPU programming knowledge.
- [ ] V3: Warnings about pitfalls (e.g., transfers, small kernels, sync points) are framed constructively, focusing on what to do.
- [ ] V4: The chapter’s voice is consistent with other chapters in the book in terms of formality and directness.
- [ ] V5: Examples and explanations avoid unnecessary hype about GPUs and focus on practical understanding.

## Reader Impact
- [ ] R1: By the end of the chapter, a reader should be able to explain in their own words why GPUs matter for modern AI workloads.
- [ ] R2: A reader can identify at least 2–3 common GPU performance pitfalls in their own code (e.g., small kernels, extra transfers).
- [ ] R3: The exercises give the reader hands-on experience moving work from CPU to GPU and timing/observing the differences.
- [ ] R4: The reader comes away with a concrete mental model of “host vs device” and understands why minimizing transfers matters.
- [ ] R5: The chapter clearly signals when it is sufficient to stay at the Python level and when it might be worth learning custom kernels.

## Chapter-Specific
- [ ] X1: The CPU vs GPU section clearly communicates the “few smart CPU cores vs many simple GPU threads” mental model, including threads, blocks, and grids.
- [ ] X2: The host–device memory section explicitly covers pinned memory, non-blocking transfers, and how they interact in a typical PyTorch `DataLoader`.
- [ ] X3: The CUDA kernel basics section ties the C++ example back to what actually happens when a Python user calls PyTorch or CuPy ops.
- [ ] X4: The performance pitfalls section maps directly to observable profiler symptoms (e.g., low utilization, many tiny kernels, memcpy spikes) and proposes practical fixes.
- [ ] X5: The exercises include at least one timing experiment that separates compute time from transfer time and highlights the impact of synchronization (`torch.cuda.synchronize()` or equivalent).
