# Setting Up Reproducible Environments

This chapter focuses on building Python environments you can trust: ones that are easy to recreate, debug, and share across laptops, servers, and clusters. You’ll see how to use virtual environments, Conda, and containers together with good dependency hygiene so that “it works on my machine” becomes “it works everywhere we care about.”

Learning objectives:
- Understand what “reproducibility” means for Python and AI workloads, and what it *doesn’t* guarantee.
- Compare virtualenv/venv, Conda/mamba, and container-based approaches, and know when to reach for each.
- Create isolated environments for CPU and GPU workloads.
- Pin and export dependencies in a way that works across machines and over time.
- Integrate environment setup with CI and basic automation so environments are recreated the same way everywhere.
- Recognize common failure modes (“it runs but is not the same”) and simple checks to catch them early.

Sections:
- Why reproducible environments matter
- Python environments 101: venv, virtualenv, and pip
- Conda and mamba: managing Python plus system libraries
- Containers (Docker/Podman) for portable runtimes
- Environment files, lockfiles, and dependency pinning
- Integrating environments with CI and teams
- Exercises: building and sharing a minimal reproducible setup
- Quick reference: choosing an environment strategy

## Why reproducible environments matter

In small experiments, it’s tempting to `pip install` whatever you need until the script finally runs. That works—once. The trouble starts when you come back a month later, move to a new machine, or hand the project to a teammate and discover that you can’t reconstruct exactly what “working” meant. A missing system library here, a silently upgraded dependency there, and suddenly your results don’t match anymore.

For AI workloads, this problem is amplified. Models are sensitive to library versions, CUDA/driver compatibility, BLAS implementations, and even minor changes in default settings. When you’re trying to compare experiments, debug performance regressions, or reproduce a paper, you need to know that the environment is the same, not just “similar enough”.

It’s helpful to be concrete. Imagine you trained a model with:

- Python 3.11.6
- PyTorch 2.2
- CUDA 12.1 runtime
- `torchvision` 0.17

If a teammate later runs inference with Python 3.12 and a newer PyTorch that defaults to a different convolution algorithm, you might see slower throughput or slightly different numerics even if the code is unchanged. Without a record of the original environment, tracking this down can take hours.

A very simple but effective “did I change something important?” check is to log a short environment summary alongside your results:

```bash
python -c "import sys, torch; print(sys.version); print('torch', torch.__version__)"
```

If the same experiment later behaves differently, you can compare these logs before digging into the rest of the stack.

Reproducible environments also save time operationally. If you can rebuild your training or inference environment from a small set of declarative files (an `environment.yml`, `requirements.txt`, or `Dockerfile`), then onboarding new machines, spinning up ephemeral CI jobs, or scaling to clusters becomes routine instead of an adventure in dependency archaeology.

Finally, good environment practices create a clean boundary between “what this project needs” and “what happens to be installed on this box”. That boundary makes your work more portable: it can move between laptops, on-prem clusters, and cloud services with far fewer surprises. The rest of this chapter is about drawing that boundary well, using the tools you already know—Python, Conda, and containers—in a more deliberate way, and setting up habits you’ll reuse when we talk about scaling, deployment, and production later in the book.

> Reproducibility here is about **software and configuration**: given the same code, data, and environment description, we should be able to recreate an equivalent environment. It does *not* by itself guarantee scientific validity or protect you from changes in external systems (APIs, datasets, hardware generations), though it makes those changes easier to detect.
>
> In particular, environment reproducibility does not:
> - Eliminate stochasticity in training (random seeds, nondeterministic kernels).
> - Freeze external data sources or APIs in time.
> - Guarantee numerical identity across different CPU/GPU models.
>
> A useful mental model is: *same code + same data + same environment description ⇒ same behavior, up to hardware-level differences*. When that doesn’t hold, you at least have a concrete place to start debugging.

## Python environments 101: venv, virtualenv, and pip

The simplest building block for reproducible Python setups is the *virtual environment*: a self-contained directory with its own Python interpreter and installed packages. Instead of installing everything into your system Python, you create one environment per project (or per use case) and let that environment hold exactly the dependencies that project needs.

Modern Python ships with `venv` in the standard library:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell/CMD
python -m pip install --upgrade pip
python -m pip install numpy torch
```

After activation, `python` and `pip` refer to the environment-local versions, not the system ones. Deactivating is just:

```bash
deactivate
```

A quick sanity check that you’re using the right environment is:

```bash
which python     # Linux/macOS
# or
where python     # Windows
```

You should see a path inside `.venv`, not a system location like `/usr/bin/python`.

For Windows users, a slightly more explicit check is:

```powershell
(.venv) PS C:\path\to\proj> python -c "import sys; print(sys.executable)"
# Expect something like: C:\path\to\proj\.venv\Scripts\python.exe
```

The older `virtualenv` tool provides similar behavior with a few extra features and historical quirks. In most modern setups, `venv` is enough; if you already use `virtualenv`, you can treat it as a drop-in alternative.

Package management in these environments is still handled by `pip`. `pip` installs from PyPI (and custom indexes), and you can capture the state of an environment with:

```bash
pip freeze > requirements.txt
```

and later recreate it with:

```bash
pip install -r requirements.txt
```

This basic pattern—*create an isolated environment, install dependencies with `pip`, then freeze them to a file*—is the foundation for everything else in this chapter. Conda, mamba, and containers add more power and portability, but they’re all solving the same core problem: “given this project, what exactly should be installed, and how do we recreate it?”.

There are two common pitfalls to keep in mind:

- Don’t `pip install` directly into your system Python “just this once”; that’s how subtle conflicts start.
- Be careful mixing `sudo` with `pip`; if you need system-wide packages, use your OS package manager instead.

A couple of lightweight habits make `venv`-based projects much easier to live with:

- Add `.venv/` to your `.gitignore` so you never commit environments.
- Put a short “how to set up the env” snippet at the top of your `README.md` and keep it in sync with your actual workflow.

If you want a minimal “getting started” recipe you can drop into a README for a simple `venv` + `pip` project, it can be as short as:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Anyone who can run these three commands (after cloning your repo) should end up with the same environment you used during development, as long as they’re on a reasonably similar OS and architecture.

If you need stricter control—for example, to ensure CI and local development use identical versions—you can combine this with a lockfile workflow (described later) so that `requirements.txt` is *generated* from a more human-friendly spec rather than edited directly.

## Conda and mamba: managing Python plus system libraries

`venv` and `pip` work well when your dependencies are mostly pure-Python wheels. As soon as you need specific CUDA versions, BLAS libraries, or non-Python tools (e.g., `ffmpeg`, `git`, compilers), life gets trickier. This is where **Conda** (and its faster drop-in replacement **mamba**) shine: they manage Python packages *and* many system-level libraries in one place.

A typical Conda workflow looks like this:

```bash
# Create an environment with a specific Python version
conda create -n myproj python=3.11
conda activate myproj

# Install core scientific stack
conda install numpy scipy pandas

# Install GPU-enabled PyTorch from the PyTorch channel (example)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

Mamba uses the same syntax but is significantly faster at solving environments:

```bash
mamba create -n myproj python=3.11
mamba activate myproj
```

On some shells, you may need to run `conda init` or `mamba init` once so that `conda activate` / `mamba activate` are available.

The key ideas:

- **Environments**: Like `venv`, Conda environments are isolated directories with their own Python and packages.
- **Channels**: Conda pulls packages from *channels* (`conda-forge`, `pytorch`, `nvidia`, etc.). Choosing good channels (often `conda-forge` + a vendor channel) simplifies dependency resolution.
- **Non-Python deps**: Conda can install things like CUDA runtimes, MKL/OpenBLAS, and CLI tools, which makes GPU setups and scientific stacks more reproducible across machines.

You can export and recreate environments declaratively:

```bash
# Export intent (packages you explicitly asked for)
conda env export --from-history > environment.yml

# Recreate elsewhere
conda env create -f environment.yml
conda activate myproj
```

If you need to update an existing environment from a tweaked `environment.yml`, you can run:

```bash
conda env update -f environment.yml --prune
```

This updates packages to match the file and prunes ones that are no longer listed, which helps avoid slow “environment drift” as you experiment.

In this book, we’ll use Conda/mamba when we need tighter control over CUDA/toolchain versions or when we want a single file (`environment.yml`) that captures both Python and key native dependencies. You can still combine Conda with `pip` inside an environment, but it’s best to keep most heavy dependencies in Conda for easier resolution and fewer binary mismatches.

A simple rule of thumb:

- Prefer Conda packages for anything with large native dependencies (NumPy, PyTorch, CUDA, compilers).
- Use `pip` inside the Conda environment for Python-only extras that aren’t well supported on Conda.

If you see cryptic “unsatisfiable dependencies” errors, check that you’re not mixing many channels arbitrarily (e.g., `defaults` + `conda-forge` + several vendors). Standardizing on a small set like `conda-forge` plus one or two vendor-specific channels usually leads to more stable solves.

A minimal debugging checklist when a Conda env “works on one machine but not another”:

- Run `conda list` on both and compare key packages.
- Confirm both machines are using the same `environment.yml` and channels.
- Check whether one machine has an extra `pip install` step that isn’t documented.
- Verify that both machines are using the same Conda implementation (e.g., both `conda` or both `mamba`) and similar versions.

If the discrepancy is still mysterious, try recreating the environment from scratch on both machines with a fresh prefix instead of updating an existing env. Hidden state from previous experiments is a frequent source of “works here but not there” confusion.

## Containers (Docker/Podman) for portable runtimes

Virtualenvs and Conda environments give you repeatability *on a given OS image*. Containers go one step further: they package your code, dependencies, and much of the underlying userland into a single artifact that you can run on laptops, servers, and clusters with minimal drift. This makes them a natural fit for CI, batch jobs, and production inference services.

At a high level, a container image is built from a recipe (a `Dockerfile`, `Containerfile`, or similar). Here’s a minimal example using Python and `pip`:

```Dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps (if needed) and Python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

You build and run it with Docker:

```bash
docker build -t myproj:latest .
docker run --rm myproj:latest
```

or with Podman (a daemonless, rootless-friendly alternative):

```bash
podman build -t myproj:latest .
podman run --rm myproj:latest
```

For GPU workloads, you typically start from a CUDA-enabled base image and enable GPU access at run time:

```bash
# Example: NVIDIA + Docker
docker run --rm --gpus all myproj:latest
```

On systems that use Podman, GPU support is configured slightly differently, but the idea is the same: the container image describes *what* to run, and the runtime (`docker`/`podman`) decides *where* it runs and which devices are visible.

Containers don’t replace virtualenvs or Conda; they often embed them. A common pattern is:

- Use Conda/mamba or `pip` inside the image to define the Python environment.
- Use the container boundary to lock down OS packages, CUDA runtimes, and CLI tools.
- Use tags (`myproj:cpu`, `myproj:gpu`) to distinguish different hardware targets.

When a containerized app “works locally but fails in CI”, environment drift is usually the culprit. A quick way to check is to:

```bash
docker run --rm -it myproj:latest python -m site
# or inspect python -c "import torch; print(torch.__version__)"
```

and compare that with the versions you expect from your `requirements.txt` / `environment.yml`.

A complementary practice is to record the image tag you used for a given experiment or deployment (for example in commit messages or experiment logs). That way, you can always pull the exact same environment again if you need to rerun or debug.

One practical gotcha: containers share the host kernel. If you upgrade the host to a new kernel or GPU driver major version, double-check that your older images still work as expected, especially for CUDA-heavy workloads.

The payoff is strong: if you can run `docker run myproj:latest` (or the equivalent Podman command) on your laptop and in CI, you have a portable, versioned environment that behaves consistently across the rest of the hardware landscape described in this book.

## Environment files, lockfiles, and dependency pinning

Creating an environment interactively is fine for exploration, but for reproducibility you need a *text description* of that environment that can live in version control. That description is usually some combination of environment files and lockfiles, plus a strategy for how tightly you pin versions.

For `pip` + `venv`, the simplest approach is:

```bash
# During development
pip install numpy torch

# Capture exact versions
pip freeze > requirements.txt
```

The resulting `requirements.txt` pins every package to an exact version (and often pulls in more transitive deps than you care about). This is good for “bit-for-bit” rebuilds but can be brittle over time. A common pattern is:

- A hand-maintained `requirements.in` (or `pyproject.toml`) with loose pins (`numpy>=1.26,<1.28`).
- A generated `requirements.txt` with strict pins, produced by a tool like `pip-tools`’s `pip-compile`.

This lets you give developers some flexibility locally (upgrade within a range) while making CI and production use a fully resolved set of versions.

A tiny example of this flow:

```bash
# requirements.in
numpy>=1.26,<1.28
torch>=2.2,<2.3

# Compile to a fully pinned requirements.txt
pip-compile requirements.in
```

For Conda/mamba, the equivalent is `environment.yml`:

```yaml
name: myproj
channels:
  - conda-forge
  - nvidia
dependencies:
  - python=3.11
  - numpy
  - pytorch
  - pytorch-cuda=12.1
  - pip
  - pip:
      - wandb>=0.16,<0.18
```

You commit this file and create the environment with:

```bash
conda env create -f environment.yml
# or
mamba env create -f environment.yml
```

Conda can also export the full, resolved environment (including exact builds) as a more “locked” spec, but those files tend to be platform-specific and noisy. In practice:

- Use a *human-edited* `environment.yml` / `requirements.in` for intent.
- Optionally generate a stricter, machine-edited lockfile for CI and production (e.g., `conda-lock` outputs, or a fully pinned `requirements.txt`).
- Expect lockfiles that include platform-specific builds to differ between Linux, macOS, and Windows.

Container images act as a kind of “binary lockfile”: the image tag (`myproj:2024-01-15`) points at an exact combination of OS + packages. You still keep your `requirements.txt` or `environment.yml` in the repo, but the built image is what you deploy.

The key habit is this: never rely on “whatever happens to be installed right now.” Always have one or more small, versioned files that answer the question, *“what environment did we mean to use for this project?”*. Those same files should be used everywhere: on your laptop, on a teammate’s machine, and inside CI. If your local setup says `pip install -r requirements.txt` or `mamba env create -f environment.yml`, your CI should be doing the exact same thing, not maintaining a separate, CI-only environment definition.

As a quick self-check, ask:

- If my laptop died, could I recreate the environment from the repo alone?
- If a bug appears only in production, do I know exactly which dependency versions prod is running?
- Do my environment files and lockfiles have clear owners, and are changes to them tested in CI?

If the honest answer is “not really,” tighten your environment files or add a lock step.

When environments must stay reproducible for years (e.g., regulatory or scientific-audit reasons), consider archiving both the text specs *and* prebuilt artifacts: exported Conda lockfiles, versioned container images, and a short “how to rebuild” note that assumes only a fresh machine and public registries.

## Integrating environments with CI and teams

A reproducible environment is most useful when everyone actually uses it: your future self, your teammates, and your CI system. That means baking environment creation into your project’s workflow instead of leaving it as a README footnote.

For a `venv` / `pip` project, your CI job might look like:

```yaml
# .github/workflows/ci.yml (excerpt)
steps:
  - uses: actions/checkout@v4

  - uses: actions/setup-python@v5
    with:
      python-version: "3.11"

  - name: Set up venv
    run: |
      python -m venv .venv
      source .venv/bin/activate
      python -m pip install --upgrade pip
      pip install -r requirements.txt
      pytest
```

For Conda/mamba-based projects:

```yaml
# .github/workflows/ci.yml (excerpt)
steps:
  - uses: actions/checkout@v4

  - uses: conda-incubator/setup-miniconda@v3
    with:
      activate-environment: myproj
      environment-file: environment.yml
      auto-activate-base: false

  - name: Run tests
    run: pytest
```

The important part is that CI creates the environment from the same files you use locally (`requirements.txt`, `environment.yml`, `Dockerfile`). If CI fails to build the environment, you know your instructions are incomplete or out of date.

For teams, a few simple habits go a long way:

- **Standardize commands**: Add a `Makefile` or small shell script with targets like `make env`, `make test`. This avoids every person inventing their own setup steps.
- **Document one canonical path**: In `README.md`, show *one* recommended way to create the environment (e.g., Conda + `environment.yml`), and clearly label anything else as optional.
- **Version and review environment files**: Treat changes to `requirements.txt`, `environment.yml`, or `Dockerfile` like code. They should go through code review and CI.
- **Reuse env definitions everywhere**: Make sure the exact same env files (`requirements.txt`, `environment.yml`, `Dockerfile`) are the ones wired into CI and deployment, so there’s no hidden “CI-only” or “prod-only` environment.
- **Automate the boring parts**: If onboarding still involves a long README section called “Environment Setup”, consider replacing it with a single command like `make dev-env` that encodes the same steps.
- **Add a quick “health check”**: A tiny script (or `make check-env`) that imports the critical libraries and prints their versions is an easy smoke test that the environment matches expectations.

A minimal `check-env.py` might look like:

```python
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

Hooking a script like this into CI catches many “it runs, but not with the environment we expected” issues before they reach production.

In later chapters, when we discuss scaling to clusters and production, we’ll assume you already have this discipline: a project is not “set up” until anyone on the team—and your CI system—can recreate its environment with one or two short commands.

## Exercises: building and sharing a minimal reproducible setup

These exercises are meant to be small and practical. Pick the tools that fit your workflow (venv, Conda, containers) and try at least one variant. As you work through them, pay attention to *exactly* which files you would commit to share the environment with someone else.

### 1. venv + pip + requirements.txt

1. In a fresh clone of this repo (or a toy project), create a virtualenv and install a couple of packages:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install numpy rich
   ```

2. Capture the environment:

   ```bash
   pip freeze > requirements.txt
   ```

3. Delete `.venv`, recreate it from `requirements.txt`, and confirm a short script runs:

   ```bash
   rm -rf .venv
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

   python -c "import numpy, rich; print(numpy.__version__, rich.__version__)"
   ```

4. (Optional) Share the repo with a teammate or run the same commands on another machine and check whether the printed versions match. If they don’t, inspect the differences in `requirements.txt`.

5. (Reflection) Note which files actually mattered for reproduction (typically: your code, `requirements.txt`, and maybe a small setup script). Imagine your future self trying this again in six months—would they have enough information?

### 2. Conda/mamba + environment.yml

1. Create a minimal `environment.yml`:

   ```yaml
   name: repro-env
   channels:
     - conda-forge
   dependencies:
     - python=3.11
     - numpy
     - pytorch
   ```

2. Build and activate the environment:

   ```bash
   mamba env create -f environment.yml  # or: conda env create -f environment.yml
   conda activate repro-env
   python -c "import numpy, torch; print('OK:', numpy.__version__, torch.__version__)"
   ```

3. Copy `environment.yml` to another machine (or container) and repeat. Confirm the same code runs without editing the file. Note any differences in package build strings (e.g., CPU vs GPU builds) and how they relate to the underlying hardware.

4. (Optional) Run `conda list --name repro-env > repro-env-spec.txt` and compare this “fully resolved” spec between machines. Decide whether you want to keep such specs under version control for critical projects.

### 3. Tiny containerized script

1. Create `requirements.txt`:

   ```text
   numpy
   ```

2. Create a minimal `Dockerfile`:

   ```Dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY main.py .
   CMD ["python", "main.py"]
   ```

3. Add a tiny `main.py`:

   ```python
   import numpy as np
   print("Sum:", np.arange(10).sum())
   ```

4. Build and run:

   ```bash
   docker build -t tiny-repro:latest .
   docker run --rm tiny-repro:latest
   ```

5. (Optional) Tag the image with a date-based tag (e.g., `tiny-repro:2024-01-15`) and note that anyone who pulls that tag is effectively using the same “locked” environment.

6. (Optional, deeper) Push the image to a registry (Docker Hub, GHCR, or your internal registry), pull it from a different machine, and confirm the output matches exactly.

As you do these exercises, keep a short list of “env description” files you ended up committing (such as `requirements.txt`, `environment.yml`, or `Dockerfile`). That inventory is often the clearest, most concrete summary of how reproducible your current workflow really is.

## Quick reference: choosing an environment strategy

When you start a new project, you don’t need to overthink environment tooling. A simple decision tree is enough:

- **Pure Python, few deps, no GPUs?**  
  Use `venv` + `pip` + `requirements.txt`.

- **Needs CUDA, BLAS, or tricky native libs, but not containers?**  
  Use Conda/mamba + `environment.yml` (with `pip` only for extras).

- **Will run in CI, on multiple machines, or in production services?**  
  Use containers with a `Dockerfile`, and inside the image use either `venv` + `pip` or Conda/mamba.

Whichever path you choose, keep the same core promises:

- The environment is **isolated** (per project, not global).
- There is a **small text description** in version control (`requirements.txt`, `environment.yml`, `Dockerfile`, or all three).
- CI and teammates use **exactly the same** description you do locally.

If you keep those promises, “it works on my machine” becomes much rarer—and when it does appear, you have the tools to track it down.
