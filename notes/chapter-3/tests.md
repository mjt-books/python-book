# Tests for chapter-3

## Structural
- [ ] S1: The chapter opens with a clear statement of its purpose and how it fits into the overall book.
- [ ] S2: The major sections (why, timing basics, CPU profiling, GPU profiling, interpreting output, benchmarking, exercises) appear in a logical progression.
- [ ] S3: All code examples are fenced with proper language tags and appear in the section that introduces or explains them.
- [ ] S4: The exercises section clearly references and builds on earlier sections in this chapter.
- [ ] S5: There are no obvious missing transitions or jumps where a new concept appears without prior context.

## Clarity
- [ ] C1: The distinctions between benchmarking, profiling, and tracing are explicitly defined in plain language.
- [ ] C2: The reader can easily tell when the text is describing CPU tools vs GPU/accelerator tools.
- [ ] C3: Each code sample has enough inline context that a reader can understand what it does without guessing.
- [ ] C4: Any advice about “warmup”, “synchronization”, and “realistic workloads” is explained in a way a non-expert can follow.
- [ ] C5: The description of how to interpret profiler tables (sorting, cumulative vs self time, hotspots) is concrete and unambiguous.

## Voice & Tone
- [ ] V1: The tone stays practical and coaching, avoiding unnecessary jargon or condescension.
- [ ] V2: The chapter emphasizes habits and workflows over clever tricks, consistent with a “hands-on engineering” voice.
- [ ] V3: Warnings about common mistakes (premature optimization, misleading microbenchmarks) are phrased as guidance, not scolding.
- [ ] V4: Examples and explanations avoid hype about tools or hardware and stay grounded in realistic workflows.

## Reader Impact
- [ ] R1: After reading, a motivated reader could set up a basic timing or `%timeit` experiment on their own code.
- [ ] R2: The reader is given at least one concrete recipe for running and inspecting a CPU profile on a Python script.
- [ ] R3: The reader is given at least one concrete recipe for running and inspecting a GPU/accelerator profile (or knows what to use if they have a GPU).
- [ ] R4: The exercises provide a clear end-to-end path from timing to profiling to making and re-measuring a change.
- [ ] R5: The chapter makes clear why profiling/benchmarking matter specifically for AI/ML workloads, not just generic Python.

## Chapter-Specific
- [ ] X1: The chapter clearly explains why timers alone are insufficient and how profiling adds “where” information beyond “how long”.
- [ ] X2: There is an explicit, actionable checklist or pattern for designing fair, repeatable benchmarks (fixed inputs, warmup, multiple runs, environment details).
- [ ] X3: GPU examples explicitly demonstrate the need for synchronization when timing and explain what happens if you omit it.
- [ ] X4: The section on interpreting profiler output shows how to pick a small number of hotspots to work on instead of trying to optimize everything.
- [ ] X5: The final exercises form a coherent mini-workflow (baseline timing → CPU profile → optional GPU profile → small change and re-measure) that mirrors the chapter’s main message.
