# Active Context: Job Description Ranker - Developer Tooling & Type Hint Fixes

**Date:** 2025-04-22 (Updated 2025-04-22 13:15 UTC)

**Current Focus:** Refining developer tooling (`justfile`) and fixing type hints.

**Recent Changes:**
- **Developer Tooling (`justfile`):**
    - Removed placeholder `main.py`.
    - Refactored `justfile` to provide comprehensive, flexible commands for running the application (`rank`), testing (`test`), type checking (`typecheck`), linting (`lint`), formatting (`format`), combined checks (`check`, `format-check`), and cache management (`clean-cache`, `clean-explanations`, `cache-stats`).
    - Commands now use `*ARGS` for flexibility and include default paths/targets.
    - Added clear descriptions for `just --list`.
- **Type Hint Fix:** Resolved `mypy` error for `sklearn.metrics.pairwise` import in `src/rank_jobs.py` by adding `# type: ignore[import-untyped]`.

**Previous Changes (Relevant):**
- **Ideal CV Prompt Refinement:** Updated `prompts/system_message.txt` for better focus.
- **Explanation Threshold & Caching:** Implemented conditional explanation generation/caching based on `explanation_threshold` in `config.yaml`.

**Next Steps:**
1.  Update `progress.md`, `systemPatterns.md`, and `techContext.md` to reflect the `justfile` changes and `main.py` removal.
2.  Review the quality of the generated explanations (if score >= threshold) and refine `prompts/explanation_prompt.txt` if necessary.
3.  Consider adding unit tests for the explanation caching functions.
4.  Discuss and prioritize next features (e.g., multi-lingual CV support, CLI/API compatibility).
5.  Update `.clinerules` with notes on using `just`.

**Open Questions/Decisions:**
- Is the default `explanation_threshold` of 0.6 appropriate? (Can be adjusted in `config.yaml`).
- Is the quality of the generated explanations acceptable? (Needs review).
