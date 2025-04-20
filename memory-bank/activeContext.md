# Active Context: Job Description Ranker - Feature Implementation

**Date:** 2025-04-20 (Updated 2025-04-20 11:05 UTC)

**Current Focus:** Feature implementation complete. Ready for end-to-end testing and review.

**Recent Changes:**
- **Project Structure:** Moved main script to `src/rank_jobs.py` and configured `pyproject.toml` for `src` layout. Installed project in editable mode (`uv pip install -e .`).
- **Configurability:**
    - Created `config.yaml` to manage paths, model names, temperature, prompt files, and cache settings.
    - Updated `src/rank_jobs.py` to load and use settings from `config.yaml`.
- **Prompt Externalization:**
    - Moved LLM system prompt for ideal CV generation to `prompts/system_message.txt`.
    - Created `prompts/explanation_prompt.txt` for match explanation.
    - Updated `src/rank_jobs.py` to load prompts from files specified in `config.yaml`.
- **Caching:**
    - Implemented file-based caching for generated ideal CVs (as `.md`) and their embeddings (as `.npy`).
    - Cache keys are generated based on JD content and relevant configuration (LLM model, embedding model, system prompt) using SHA256.
    - Cache files are stored in subdirectories based on the first two characters of the hash key (e.g., `cache/ab/abcdef..._ideal_cv.md`).
    - Added cache configuration (`enabled`, `directory`) to `config.yaml`.
    - Updated `src/rank_jobs.py` with functions (`get_cache_key`, `get_cache_paths`, `load_from_cache`, `save_to_cache`) and integrated logic into the main processing loop.
- **Testing:**
    - Added `pytest` and necessary type stubs (`data-science-types`, `types-PyYAML`) as dev dependencies.
    - Created `tests/` directory.
    - Implemented unit tests for `load_config` in `tests/test_config.py`.
    - Implemented unit tests for caching functions in `tests/test_cache.py` using mocking.
- **Explainability:**
    - Added `generate_explanation` function to `src/rank_jobs.py` using the LLM and the new explanation prompt.
    - Integrated explanation generation into the main loop.
    - Updated the final output format to include the generated explanation for each ranked job.

**Next Steps:**
1.  Run the script end-to-end (`python src/rank_jobs.py`) to test the integrated features (requires `OPENAI_API_KEY` in `.env`).
2.  Review the output, including the generated explanations.
3.  Update `progress.md` and `systemPatterns.md` to reflect the new architecture and features.
4.  Consider adding more comprehensive tests (e.g., integration tests mocking external APIs).

**Open Questions/Decisions:**
- Does the user have a valid `OPENAI_API_KEY` in the `.env` file? (Still required for full run).
- Is the quality of the generated explanations acceptable?
