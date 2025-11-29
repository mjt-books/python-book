# Tests for chapter-2

## Structural
- [ ] S1: The chapter begins with a short orientation (overview + learning objectives) that makes its scope and role in the book clear.
- [ ] S2: All major tools/approaches (venv/pip, Conda/mamba, containers, env files/lockfiles, CI integration, exercises) are covered in clearly labeled sections that match the outline.
- [ ] S3: Code blocks are consistently formatted with appropriate languages (bash, Dockerfile, yaml, python, text) and are separated from surrounding prose with blank lines.
- [ ] S4: The exercises section is clearly separated from the main explanatory content and contains numbered sub-exercises with step-by-step commands.
- [ ] S5: Any cross-references to other chapters or later material are forward-compatible (no broken promises about concepts not actually covered in the book).

## Clarity
- [ ] C1: The chapter defines “reproducible environments” in concrete, non-jargony terms and ties the concept directly to Python/AI workflows.
- [ ] C2: The differences between venv/virtualenv/pip, Conda/mamba, and containers are explained in a way that a motivated but non-expert reader can restate in their own words.
- [ ] C3: Each example workflow (venv, Conda, containers, CI) includes enough commands that a reader could follow them without guessing missing steps.
- [ ] C4: The discussion of environment files and lockfiles clearly distinguishes “intent” files from machine-generated lockfiles without overloading the reader with tooling details.
- [ ] C5: Any mentions of GPU/CUDA dependencies are explained at a high level so that readers know why these need extra care, even if they have not used GPUs before.

## Voice & Tone
- [ ] V1: The tone is pragmatic and reassuring, treating environment management as an important but solvable engineering practice rather than as intimidating “ops magic.”
- [ ] V2: The chapter uses “you” and “we” consistently to create a collaborative, instructive tone without slipping into chatty or informal asides.
- [ ] V3: Explanations avoid unnecessary snark about tools or workflows and instead frame trade-offs in neutral, engineering terms.
- [ ] V4: Warnings about pitfalls (e.g., silent upgrades, system package drift) are direct but not alarmist, staying focused on actionable guidance.
- [ ] V5: The narrative reinforces the book’s overall theme of building trustworthy, maintainable AI systems rather than treating environments as an isolated topic.

## Reader Impact
- [ ] R1: A reader who follows the chapter should be able to set up, tear down, and recreate a basic venv-based environment from a requirements file on a fresh machine.
- [ ] R2: A reader should come away with at least one concrete “single source of truth” pattern (e.g., environment.yml, requirements.in + requirements.txt, Dockerfile) they could adopt in their own projects.
- [ ] R3: The chapter explicitly connects environment reproducibility to downstream tasks like CI, collaboration with teammates, and deployment, not just solo experimentation.
- [ ] R4: The exercises are realistic in scale (doable in under an hour) and clearly reinforce the main concepts rather than introducing unrelated complexity.
- [ ] R5: The chapter gives the reader at least one simple “getting started” recipe they could paste into a README for a new project.

## Chapter-Specific
- [ ] X1: The chapter clearly articulates why environment drift (e.g., upgraded dependencies, missing system libraries, CUDA mismatches) harms reproducibility for AI workloads in particular.
- [ ] X2: The Conda/mamba section explicitly highlights their strength in managing non-Python and GPU-related dependencies, with at least one concrete CUDA or BLAS example.
- [ ] X3: The containers section demonstrates a full minimal workflow (Dockerfile + build + run) and explains how containers complement rather than replace venv/Conda.
- [ ] X4: The environment files/lockfiles section makes an explicit distinction between human-maintained spec files and machine-generated, pinned files, and ties this to CI usage.
- [ ] X5: The CI integration examples show how the same environment definitions used locally (requirements.txt, environment.yml, Dockerfile) are reused in CI steps to prevent configuration drift.
