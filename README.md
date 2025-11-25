# A book on Python

A practical, professional — and occasionally playful — guide to using Python and AI to harness multiple hardware resources. This book contains example code, hands-on recipes, and 20 chapters focused on scaling AI across CPUs, GPUs, TPUs, edge devices, and clusters.

Key points:
- Example-rich, runnable code
- Tone: professional and fun
- 20 focused chapters
- Primary focus: using AI to utilize multiple hardware resources efficiently and reproducibly

## How to use this book
Each chapter contains conceptual explanations, code examples, profiling tips, and end-of-chapter exercises. Most examples assume a Linux environment and include commands for common tools.

## Chapters

1. [Foundations: AI, Python, and the hardware landscape](chapters/chapter-1.md)  
2. [Setting up reproducible environments (venv, conda, containers)](chapters/chapter-2.md)  
3. [Profiling and benchmarking: measuring performance and bottlenecks](chapters/chapter-3.md)  
4. CPU optimizations and multi-threading (vectorization, OpenMP, asyncio)  
5. GPU fundamentals and CUDA basics for Python users  
6. Using cuDNN, cuBLAS, and vendor libraries effectively  
7. Mixed precision, quantization, and numerical trade-offs  
8. Data parallelism: batch sharding and distributed dataloaders  
9. Model parallelism: slicing large models across devices  
10. Multi-node training and parameter servers  
11. TPUs and accelerator-specific patterns (JAX/TPU tips)  
12. Orchestration and scheduling with Kubernetes and Slurm  
13. Ray, Dask, and distributed compute frameworks for Python  
14. Edge deployment: ONNX, TensorRT, and inference on constrained devices  
15. FPGA and specialized accelerators: when and how to use them  
16. Memory management, checkpointing, and fault tolerance  
17. Energy efficiency and cost-aware training strategies  
18. Real-world case studies: scaling transformer models end-to-end  
19. Tools, utilities, and CI for training pipelines (logging, metrics, tracing)  
20. Appendix: debugging, useful libraries, and further reading