# Conventions Used in This Book

This page explains how we format code, commands, UI elements, warnings, and file references.

## Code and Monospaced Text

- Inline code uses backticks: `torch.cuda.is_available()`
- Code blocks use fenced code with a language hint:

```python
# Example: simple CUDA check
import torch

print(torch.cuda.is_available())
```

- Configuration snippets or non-Python text use an appropriate language (e.g. `bash`, `yaml`, `json`) or plain fences.

## UI Elements and Keyboard Shortcuts

- **UI elements** (buttons, menu items, labels) are in **bold**:  
  “Click **Run**, then **View ▸ Logs**.”
- **Keyboard shortcuts** use `+` between keys:  
  `Ctrl+C`, `Shift+Enter`, `Cmd+K Cmd+S` (for chorded shortcuts).

## Shell Prompts and Commands

- Shell commands are shown with a prompt, but you should type only what follows the prompt:
  - `$` for non-root user (most examples)
  - `#` for root user or elevated shells

```bash
# Non-root example
$ python -m venv .venv

# Root example
# apt-get update
```

- Multi-line commands use `\` for line continuation:

```bash
$ pip install \
    torch \
    torchvision
```

- Output from commands is either shown below the command or marked with comments:

```bash
$ nvidia-smi
# ...tool output...
```

## Warnings, Notes, and Tips

We use consistent callouts to highlight important information. The exact rendering depends on your viewer, but the patterns are:

> **Warning**  
> Be explicit when running commands as `root`. A wrong flag can affect your whole system.

> **Note**  
> When we say “Linux”, we typically assume a recent Ubuntu or Debian-based distribution unless otherwise specified.

> **Tip**  
> If a command fails, re-run it with `-h` or `--help` to inspect available options and defaults.

In source Markdown, these are written as blockquotes:

```markdown
> **Warning**  
> Text of the warning.

> **Note**  
> Text of the note.

> **Tip**  
> Text of the tip.
```

Some chapters may use similar boxed callouts offered by tools like Sphinx or MkDocs; the semantics are the same.

## Filenames vs Paths

We distinguish between bare filenames and full/relative paths:

- **Filenames** use inline code: `train.py`, `requirements.txt`.
- **Paths** include directory components and also use inline code:
  - Relative paths: `src/train.py`, `configs/exp01.yaml`
  - Absolute paths: `/home/user/projects/`, `/data/checkpoints/`

When we refer to a *directory*:

- Trailing slash is optional but sometimes used for clarity:
  - “Create `logs/` if it does not exist.”
  - “Place your data in the `data` directory.”

If a command is meant to be run from a specific directory, we state it explicitly:

> **Note**  
> Run all commands in this chapter from the `project-foo/` root unless we say otherwise.

## Environment and Platform-Specific Notation

Occasionally we tag commands or snippets with the relevant environment:

- `(Linux)` Commands expected to work on common Linux distros.
- `(Windows)` Commands using PowerShell or `cmd.exe`.
- `(macOS)` macOS-specific steps.

Example:

```bash
# (Linux/macOS)
$ python -m venv .venv
```

If a step differs across platforms, the chapter will provide separate blocks or subsections per platform.

