# Active Context: Job Description Ranker - Refinement & Explanation Caching

**Date:** 2025-04-21 (Updated 2025-04-21 22:54 UTC)

**Current Focus:** Implementing explanation caching and refining prompts.

**Recent Changes:**
- **Ideal CV Prompt Refinement:** Updated `prompts/system_message.txt` to explicitly instruct the LLM to exclude placeholder personal details and generic CV sections, focusing solely on extracting requirements from the JD. Tested and confirmed improved output quality.
- **Explanation Threshold:** Added `explanation_threshold` setting to `config.yaml` (default 0.6) to control when explanations are generated/cached. Updated `load_config` in `src/rank_jobs.py` to validate this setting.
- **Explanation Caching:**
    - Added new caching functions (`get_explanation_cache_key`, `get_explanation_cache_path`, `load_explanation_from_cache`, `save_explanation_to_cache`) to `src/rank_jobs.py`.
    - The explanation cache key depends on the ideal CV's cache key, the user CV text, the explanation prompt text, and relevant LLM config.
    - Modified the main loop in `src/rank_jobs.py` to check the similarity score against the `explanation_threshold`.
    - Explanations are now loaded from/saved to cache only if the score meets the threshold. If below threshold, a "skipped" message is stored.

**Next Steps:**
1.  Update `progress.md` and `systemPatterns.md` to reflect the new explanation caching logic and threshold.
2.  Review the quality of the generated explanations (if score >= threshold) and refine `prompts/explanation_prompt.txt` if necessary.
3.  Consider adding unit tests for the new explanation caching functions.
4.  Discuss and prioritize next features (e.g., multi-lingual CV support, CLI/API compatibility).

**Open Questions/Decisions:**
- Is the default `explanation_threshold` of 0.6 appropriate? (Can be adjusted in `config.yaml`).
- Is the quality of the generated explanations acceptable? (Needs review).
