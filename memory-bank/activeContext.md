# Active Context: Job Description Ranker - Initial Setup

**Date:** 2025-04-18 (Updated 2025-04-18 21:57 UTC)

**Current Focus:** Core Logic Implementation.

**Recent Changes:**
- Project initialized (Memory Bank, `.clinerules`, data files, `.gitignore`, dependencies installed).
- Created main script `rank_jobs.py`.
- Added initial structure to `rank_jobs.py`:
    - Imports
    - Constants (file paths, model names)
    - Functions for API key loading, model initialization, text loading.
    - Core logic functions (ideal CV generation, embedding, similarity calculation).
    - Main execution block (`if __name__ == "__main__":`) orchestrating the workflow.
- Updated `systemPatterns.md` to reflect `rank_jobs.py` structure.

**Next Steps:**
1.  Ensure the user has created and populated the `.env` file with `OPENAI_API_KEY`. (Cannot proceed without this).
2.  Run the script (`python rank_jobs.py`) to test the initial execution flow. This will likely trigger the download of the `sentence-transformers` model and test the API connection.
3.  Debug any errors encountered during the initial run.
4.  Refine the LLM prompt in `generate_ideal_cv` if needed based on initial output quality.
5.  Update `progress.md`.

**Open Questions/Decisions:**
- Does the user have a valid `OPENAI_API_KEY` in the `.env` file? The script will fail without it.
