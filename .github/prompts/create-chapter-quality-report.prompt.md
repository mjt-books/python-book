---
name: create-chapter-quality-report
description: Evaluate the active chapter against its tests.md and produce a quality-report.md
agent: edit
---

You are acting as a test runner for a single chapter of my book.

**Context and conventions**

- The active file path is `${file}`.
- Assume the active file is a chapter Markdown file inside the `chapters/` folder.
- The chapter identifier comes from `${fileBasenameNoExtension}`.
  - Example: if `${fileBasenameNoExtension}` is `chapter-7`, then:
    - The chapter source is `chapters/chapter-7.md`.
    - The tests file is `notes/chapter-7/tests.md`.
    - The quality report output path is `notes/chapter-7/quality-report.md`.

Use:
- The test definitions in [tests.md](notes/${fileBasenameNoExtension}/tests.md).
- The full chapter text in the active file [chapter](${file}).
- The overall book context in [README.md](README.md).

**Goal**

Generate the complete contents for a new file:

- `notes/${fileBasenameNoExtension}/quality-report.md`

This file is a structured evaluation of each test defined in `notes/${fileBasenameNoExtension}/tests.md`.

---

### Requirements for the output

1. Output MUST be valid Markdown only. Do **not** include explanations outside the file content.

2. Preserve the same top-level test sections and test IDs that appear in `tests.md`.
   - For example, if `tests.md` has sections like `## Structural`, `## Clarity`, etc., mirror those section headings.
   - For each test bullet like:
     - `- [ ] S1: Opening clearly states the chapter’s promise.`
     create a corresponding evaluation entry in the quality report.

3. For each test:
   - Mark it as:
     - `✔️ Pass`
     - `❌ Fail`
     - `⚠️ Partial`
   - Provide 1–2 sentences explaining *why* it passed or failed.
   - If the test is `❌` or `⚠️`, add a short, concrete suggestion for improvement.

4. Use this general structure, adapting section headings and test IDs to match `tests.md`:

    # Quality Report for ${fileBasenameNoExtension}

    ## Structural
    - S1: ✔️ / ❌ / ⚠️ — short explanation  
      - Suggestion: ... (only if ❌ or ⚠️)
    - S2: ...
    - S3: ...

    ## Clarity
    - C1: ...
    - C2: ...
    - C3: ...

    ## Voice & Tone
    - V1: ...
    - V2: ...
    - V3: ...

    ## Reader Impact
    - R1: ...
    - R2: ...
    - R3: ...

    ## Chapter-Specific
    - X1: ...
    - X2: ...
    - X3: ...

    ---

    ## Summary of Failing or Partial Tests
    - S2 — brief explanation
    - C3 — brief explanation
    - X1 — brief explanation
    (list all ❌ and ⚠️ tests here)

5. Do **not** rewrite or modify the chapter text in this file.
   - Only evaluate the tests and suggest improvements at a high level.

6. Do **not** include any commentary outside the Markdown file content.

Return ONLY the final Markdown for `notes/${fileBasenameNoExtension}/quality-report.md`.
