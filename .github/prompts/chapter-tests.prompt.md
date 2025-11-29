---
name: chapter-tests
description: Create or update tests.md for the active chapter file
agent: edit
---

You are helping me build a unit-test-style quality spec for a single chapter of my book.

**Context and conventions**

- The active file path is `${file}`.
- Assume the active file is a chapter Markdown file inside the `chapters/` folder.
- The chapter identifier comes from `${fileBasenameNoExtension}`.
  - Example: if `${fileBasenameNoExtension}` is `chapter-7`, then:
    - The chapter source is `chapters/chapter-7.md`.
    - The tests file should be `notes/chapter-7/tests.md`.

Use:
- The overall book context in [README.md](README.md).
- The full chapter text in the active file `${file}`.

**Goal**

Generate the complete contents for a new file:

- `notes/${fileBasenameNoExtension}/tests.md`

This file defines chapter-specific "tests" (quality checks) for *this* chapter.

**Requirements for the output**

1. Output MUST be valid Markdown only. Do **not** include explanations outside the file content.

2. Use this structure (adapting the heading to the actual chapter):

    # Tests for ${fileBasenameNoExtension}

    ## Structural
    - [ ] S1: ...
    - [ ] S2: ...
    - [ ] S3: ...

    ## Clarity
    - [ ] C1: ...
    - [ ] C2: ...
    - [ ] C3: ...

    ## Voice & Tone
    - [ ] V1: ...
    - [ ] V2: ...
    - [ ] V3: ...

    ## Reader Impact
    - [ ] R1: ...
    - [ ] R2: ...
    - [ ] R3: ...

    ## Chapter-Specific
    - [ ] X1: ...
    - [ ] X2: ...
    - [ ] X3: ...

3. For each test:
   - Make it a yes/no style check (clearly passes or fails).
   - Base it on BOTH:
     - the chapter’s role in the book overall (from README.md), and
     - the actual content and intent of the active chapter file.
   - Keep each test 1–2 lines, written in plain language.

4. In the “Chapter-Specific” section, add 2–5 tests that are unique to this chapter’s purpose:
   - Specific promises it makes,
   - Key ideas it must successfully convey,
   - Any critical examples or arguments that must land well.

5. Do NOT rewrite or critique the chapter in this file.
   Only define the tests.

Return ONLY the finished Markdown for `notes/${fileBasenameNoExtension}/tests.md`.
