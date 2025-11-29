# Tests for chapter-6

## Structural
- [ ] S1: The chapter opens with a clear explanation of what vendor libraries are and why they matter before diving into specific libraries.
- [ ] S2: cuBLAS/GEMM, cuDNN/convolutions, framework integration, tuning knobs, limitations, and exercises each have their own clearly labeled sections.
- [ ] S3: Code blocks and text flow in an order that mirrors the learning objectives listed in the chapter introduction.
- [ ] S4: Exercises appear at the end of the chapter and explicitly build on concepts introduced earlier sections.
- [ ] S5: All subsections use consistent Markdown heading levels and list formatting.

## Clarity
- [ ] C1: The roles of cuBLAS, cuDNN, and “vendor libraries” are defined in plain language without assuming prior systems knowledge.
- [ ] C2: The relationship between high-level Python ops (e.g., `x @ w`, `Conv2d`) and underlying vendor-library calls is explained with at least one concrete example per library.
- [ ] C3: The distinctions between “shape”, “layout”, and “dtype” are clearly stated and used consistently throughout.
- [ ] C4: The speed vs determinism vs memory trade-offs for the tuning knobs are stated explicitly and without ambiguous phrasing.
- [ ] C5: Any framework-specific configuration examples (PyTorch, TensorFlow, JAX) include just enough context that a reader can copy–paste and run them.

## Voice & Tone
- [ ] V1: The tone remains practical and reassuring, emphasizing what readers can control from Python without overhyping low-level details.
- [ ] V2: The chapter avoids framework tribalism and presents PyTorch, TensorFlow, and JAX even-handedly.
- [ ] V3: Jargon such as “GEMM”, “autotuning”, “workspace”, and “deterministic” is either defined on first use or made clear from context.
- [ ] V4: The narrative consistently reinforces the idea of “getting performance for free” from vendor libraries instead of shaming readers for not writing CUDA.
- [ ] V5: Warnings about pitfalls (e.g., non-determinism, memory use) are stated calmly, with actionable guidance rather than alarmist language.

## Reader Impact
- [ ] R1: After reading, a practitioner should be able to explain in 2–3 sentences how vendor libraries fit into their deep learning stack.
- [ ] R2: The reader can identify at least two concrete changes (e.g., enabling cuDNN benchmarking, adjusting dtypes) they could try in their own code.
- [ ] R3: The chapter leaves readers confident that they can get most GPU performance without writing custom kernels, while understanding when that might change.
- [ ] R4: The exercises are realistically runnable on a single-GPU setup and clearly state any hardware assumptions.
- [ ] R5: The chapter motivates profiling and inspection of kernels as a useful skill without requiring deep profiler expertise.

## Chapter-Specific
- [ ] X1: The chapter clearly explains GEMM/cuBLAS as the central building block of modern deep learning and ties this back to common layers like Linear and attention.
- [ ] X2: The cuDNN section explicitly connects convolution hyperparameters (kernel size, stride, padding, layout) to algorithm selection and performance.
- [ ] X3: The framework integration section includes at least one diagram or textual stack breakdown that shows how Python code reaches vendor libraries.
- [ ] X4: The tuning knobs section names and demonstrates the key PyTorch flags for cuDNN benchmarking and determinism, and briefly indicates TF/JAX analogues.
- [ ] X5: The “when vendor libraries fall short” section lists practical, escalating options (rewrite using standard ops, batch/fuse, use JIT/compilers, then custom kernels) in a clearly prioritized order.
