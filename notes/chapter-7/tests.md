# Tests for Chapter 7 – Make It Smaller

## Structural
- [ ] S1: The chapter opens by clearly stating the problem of oversized, unwieldy tests and why “smaller” matters.
- [ ] S2: The progression from problem → principles → concrete techniques is easy to follow without backtracking.
- [ ] S3: Sections on refactoring tests, extracting helpers, and reducing fixtures are clearly separated with headings or signposts.
- [ ] S4: Any lists of techniques (e.g., ways to shrink tests) are presented as explicit bullets or clearly enumerated items.
- [ ] S5: The chapter ends with a short, concrete summary or checklist that reinforces the “make it smaller” practices.

## Clarity
- [ ] C1: “Small” is explicitly defined (e.g., lines of code, number of concepts, setup size) rather than left vague.
- [ ] C2: Each key term (e.g., fixture, helper, abstraction boundary) is either defined or used in a way that’s self-explanatory from context.
- [ ] C3: Example tests that start “too big” and end “smaller” are shown side‑by‑side or in a clearly connected before/after sequence.
- [ ] C4: The reader can tell when the chapter is giving a rule of thumb versus a hard rule (e.g., “try to” vs “must”).
- [ ] C5: Any trade‑offs (e.g., too many helpers causing indirection) are briefly explained so the reader knows what to watch for.

## Voice & Tone
- [ ] V1: The tone stays practical and encouraging, avoiding shaming readers for having large or messy tests today.
- [ ] V2: The chapter speaks directly to the reader (“you”) and maintains a conversational, non‑academic style.
- [ ] V3: The author acknowledges real‑world constraints (legacy code, time pressure) rather than assuming a greenfield ideal.
- [ ] V4: Humor or lightness, if used, supports understanding rather than distracting from the technical guidance.

## Reader Impact
- [ ] R1: After reading, a mid‑level engineer should be able to name at least 3 concrete ways to make an existing large test smaller.
- [ ] R2: The chapter gives at least one “start tomorrow” style suggestion that feels immediately applicable in a real codebase.
- [ ] R3: The reader is warned about common failure modes (e.g., over‑abstracting helpers, hiding important details) and how to avoid them.
- [ ] R4: The reader can imagine applying the techniques within their current team’s test suite, not only in idealized examples.

## Chapter-Specific
- [ ] X1: The chapter clearly connects “small tests” to the larger book themes of maintainability, test readability, and long‑term quality.
- [ ] X2: At least one extended example walks through shrinking a specific test from bloated to focused, explaining each step’s intent.
- [ ] X3: The relationship between making tests smaller and improving failure messages/debuggability is called out explicitly.
- [ ] X4: The chapter distinguishes between making tests smaller by removing noise vs. accidentally reducing real coverage.
- [ ] X5: The closing section makes an explicit promise or invitation about how adopting “smaller tests” will change the reader’s day‑to‑day work.
