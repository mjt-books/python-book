# Tests for chapter-4

## Structural
- [ ] S1: The chapter opens with a clear explanation of why CPU optimization matters in the context of an accelerator-focused ML/AI stack.
- [ ] S2: The main sections follow a logical progression from concepts (vectorization) to mechanisms (threads/processes/async) to measurement and exercises.
- [ ] S3: Each major tool or technique (vectorization, threading, processes, native libraries, asyncio, measurement) has at least one self-contained example code block.
- [ ] S4: The exercises section clearly references and builds on techniques introduced earlier in the chapter.
- [ ] S5: Section headings match the learning objectives listed at the start (e.g., vectorization, multi-threading, async I/O, measurement).

## Clarity
- [ ] C1: The explanation of the relationship between Python loops, NumPy vectorization, and SIMD is understandable without requiring low-level CPU background.
- [ ] C2: The differences between threads and processes in Python (including the GIL) are explained in simple, non-jargony language.
- [ ] C3: The role of native libraries (BLAS, OpenMP-backed code) is described clearly enough that a reader can tell when they are "already getting" multi-threading.
- [ ] C4: The examples using `ThreadPoolExecutor`, `ProcessPoolExecutor`, and `asyncio` each state what kind of workload they are meant for (CPU-bound vs I/O-bound).
- [ ] C5: The description of measuring CPU utilization and scaling (including Amdahl’s law) is concrete enough that a reader can apply it to their own code.

## Voice & Tone
- [ ] V1: The chapter maintains a pragmatic, practice-oriented tone that focuses on what the reader should *do* rather than deep theory.
- [ ] V2: The language stays consistent with the rest of the book (direct, conversational, no unnecessary hype or alarmism about performance).
- [ ] V3: Warnings about pitfalls (GIL limits, over-threading BLAS, overhead of processes) are framed as helpful guidance, not as discouraging or intimidating.
- [ ] V4: First- and second-person usage ("you", "we") is consistent and used to keep the reader engaged and guided.
- [ ] V5: The chapter avoids unexplained acronyms or, where they appear (SIMD, BLAS, OpenMP, GIL), they are briefly expanded or contextualized.

## Reader Impact
- [ ] R1: By the end of the chapter, a reader should be able to identify at least one place in their own pipeline where NumPy-style vectorization could replace Python loops.
- [ ] R2: The reader is given clear guidance on when to choose threads vs processes vs async I/O for a given kind of workload.
- [ ] R3: The chapter makes it explicit how CPU optimization connects to overall GPU/accelerator utilization and cost (e.g., not leaving accelerators idle).
- [ ] R4: The measurement examples (e.g., with `top`/`htop` and simple timing loops) are actionable and easy to replicate on a typical dev machine.
- [ ] R5: The exercises encourage the reader to actually run and compare timings, not just read the code.

## Chapter-Specific
- [ ] X1: The "Why CPU optimization still matters in an accelerator world" section explicitly ties CPU bottlenecks to symptoms like low GPU utilization and data loader issues.
- [ ] X2: The vectorization section includes at least one example that replaces a nested Python loop with a NumPy expression and explains *why* it is faster.
- [ ] X3: The discussion of threads vs processes explicitly mentions the GIL and clearly states that CPU-bound pure-Python code does not scale well with threads.
- [ ] X4: The native libraries section explicitly warns about "double parallelism" (e.g., N Python workers each calling N-threaded BLAS) and suggests how to avoid it.
- [ ] X5: The async I/O section makes clear that `asyncio` is mainly for I/O-bound work and suggests how to combine it with thread/process pools for CPU-heavy parts.
- [ ] X6: The measurement and scaling section connects observed scaling behavior back to Amdahl’s law and explains what to do when adding workers stops helping.
