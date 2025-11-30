# Python For AI

A practical, professional (and occasionally playful), example-driven guide to using Python for AI across CPUs, GPUs, TPUs, edge devices, and clusters. Packed with runnable code, hands-on recipes, and 20 focused chapters on reproducible environments, performance optimization, and scalable training and inference.

Key points:
- Example-rich, runnable code
- Tone: professional and fun
- 20 focused chapters
- Primary focus: using AI to utilize multiple hardware resources efficiently and reproducibly

## How to use this book

Each chapter contains conceptual explanations, code examples, profiling tips, and end-of-chapter exercises. Most examples assume a Linux environment and include commands for common tools.

For detailed information on:

- Python and library versions
- Hardware assumptions (CPU/GPU/memory)
- Installation and environment setup
- Repo layout and reproducibility practices
- How examples are structured

see the dedicated guide:  
[How to Use This Book](chapters/how-to-use.md)

For how we format code, shell commands, warnings/notes/tips, and file references, see:  
[Conventions Used in This Book](chapters/conventions.md)

## Chapters

0. [Preface](chapters/preface.md)  
1. [Foundations: AI, Python, and the hardware landscape](chapters/chapter-1.md)  
2. [Setting up reproducible environments (venv, conda, containers)](chapters/chapter-2.md)  
3. [Profiling and benchmarking: measuring performance and bottlenecks](chapters/chapter-3.md)  
4. [CPU optimizations and multi-threading (vectorization, OpenMP, asyncio)](chapters/chapter-4.md)  
5. [GPU fundamentals and CUDA basics for Python users](chapters/chapter-5.md)  
6. [Using cuDNN, cuBLAS, and vendor libraries effectively](chapters/chapter-6.md)  
7. [Mixed precision, quantization, and numerical trade-offs](chapters/chapter-7.md)  
8. [Data parallelism: batch sharding and distributed dataloaders](chapters/chapter-8.md)  
9. [Model parallelism: slicing large models across devices](chapters/chapter-9.md)  
10. [Multi-node training and parameter servers](chapters/chapter-10.md)  
11. [TPUs and accelerator-specific patterns (JAX/TPU tips)](chapters/chapter-11.md)  
12. [Orchestration and scheduling with Kubernetes and Slurm](chapters/chapter-12.md)  
13. [Ray, Dask, and distributed compute frameworks for Python](chapters/chapter-13.md)  
14. [Edge deployment: ONNX, TensorRT, and inference on constrained devices](chapters/chapter-14.md)  
15. [FPGA and specialized accelerators: when and how to use them](chapters/chapter-15.md)  
16. [Memory management, checkpointing, and fault tolerance](chapters/chapter-16.md)  
17. [Energy efficiency and cost-aware training strategies](chapters/chapter-17.md)  
18. [Real-world case studies: scaling transformer models end-to-end](chapters/chapter-18.md)  
19. [Tools, utilities, and CI for training pipelines (logging, metrics, tracing)](chapters/chapter-19.md)  
20. [Appendix: debugging, useful libraries, and further reading](chapters/chapter-20.md)