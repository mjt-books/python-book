# How to Use This Book

This page explains the practical setup and assumptions behind the code and experiments in this book.

## Python version used

- Target: **Python 3.11**
- Minimum: **Python 3.10**
- Recommended: Use the **latest 3.11.x** available in your OS/package manager.

Most examples will work on Python 3.10+, but all testing is done against 3.11.

## Library versions

Core ecosystem (typical baseline):

- **NumPy**: `>=1.26`
- **SciPy**: `>=1.11`
- **Pandas**: `>=2.1`
- **PyTorch**: `>=2.2` (for GPU/TPU chapters that use PyTorch)
- **TensorFlow**: `>=2.16` (only in chapters where explicitly mentioned)
- **JAX**: `>=0.4.30` (for TPU/accelerator-specific examples)
- **CUDA toolkit**: `>=12.0` (where GPU/CUDA is required)
- **CuDNN / CuBLAS**: Version compatible with your CUDA install
- **Ray**: `>=2.9`
- **Dask**: `>=2024.1.0`

## Hardware assumptions (CPU/GPU/memory)

The book is written to be **progressive**: you can run a substantial portion on a laptop, and more advanced parts on workstations or clusters.

Baseline assumptions:

- **CPU**:
  - 4–8 cores, 64-bit (x86-64 or ARM64)
  - AVX2 support recommended for best performance
- **Memory (RAM)**:
  - Minimum: **8 GB**
  - Comfortable: **16–32 GB** for larger examples and profiling
- **GPU** (for GPU chapters):
  - NVIDIA GPU with **CUDA 12+** support and at least **8 GB VRAM**
  - Many examples will run on 6 GB, but batch sizes may need to be reduced
- **TPU / Cloud accelerators**:
  - Assumed available via cloud providers (e.g., GCE, Colab, or internal infrastructure)
  - The TPU chapters assume you can provision a TPU v3 or later
- **Disk**:
  - SSD recommended
  - ~10–20 GB free space for models, datasets, and logs

Each chapter indicates if GPU/TPU is **required**, **optional but recommended**, or **not needed**.

## Installation basics

You can follow any of the environment approaches from Chapter 2, but the minimal, generic flow looks like:

1. **Create and activate a virtual environment** (example with `venv`):

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows PowerShell
   ```

2. **Upgrade pip**:

   ```bash
   pip install --upgrade pip
   ```

3. **Install base dependencies** (illustrative):

   ```bash
   pip install -r requirements.txt
   ```

   or install chapter-specific requirements, such as:

   ```bash
   pip install -r requirements-ch05-gpu.txt
   ```

4. **GPU/accelerator setup**:
   - Install appropriate **GPU drivers** for your OS.
   - Install a **CUDA-enabled build** of PyTorch/TensorFlow/JAX as directed in their official docs.
   - Verify with a simple snippet, e.g. for PyTorch:

     ```python
     import torch
     print(torch.cuda.is_available())
     ```

Throughout the book, installation commands are prefixed with the shell type (`bash`, `pwsh`) when necessary.

## Notes on environment reproducibility

This book emphasizes **reproducible experiments**:

## How examples are structured

Most chapters follow a consistent structure:

1. **Conceptual introduction**  
   Briefly explains the core idea (e.g., data parallelism, profiling, mixed precision), with diagrams where helpful.

2. **Minimal runnable example**  
   - A short script or notebook you can run end-to-end in under a minute.
   - Designed to be readable and easy to modify.

3. **Scaling up**  
   - Take the minimal example and:
     - Run it across multiple CPUs/GPUs.
     - Add profiling instrumentation.
     - Introduce fault tolerance or checkpointing, etc.

4. **Performance and profiling section**  
   - Uses tools such as:
     - `time`, `perf`, `nsys`, `nvprof`, PyTorch/TensorFlow profilers, Ray dashboard, Dask dashboard.
   - Shows how to interpret the outputs and adjust code.

5. **Variations and “what if” experiments**  
   - Change batch sizes, precision (FP32 vs mixed precision), number of workers, sharding strategy, etc.
   - Discuss trade-offs in throughput, latency, and cost.

6. **Exercises**  
   - Short tasks to:
     - Modify the example.
     - Plug in your own model or dataset.
     - Compare configurations and log results.

### Conventions used

- **Code listings** are in Python unless otherwise noted.
- Shell commands are shown with a prompt indicator, e.g.:

  ```bash
  $ python train.py --config configs/baseline.yaml
  ```

- Environment variables are written like `CUDA_VISIBLE_DEVICES=0,1`.
- File paths in examples assume a Linux-like filesystem (`/home/user/...`). Adapt to your platform as needed.

If you follow this page before diving into Chapter 1, you should be able to run almost all examples with minimal friction.
