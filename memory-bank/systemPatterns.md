# System Patterns: Job Description Ranker

**Date:** 2025-04-22 (Updated 2025-04-22 13:17 UTC)

**1. Overall Architecture:**
Command-Line Application (Python Script: `src/rank_jobs.py`) using a `src` layout. Tasks (running, testing, linting, formatting, cache management) are managed via `justfile`.

**Workflow:**
1.  **Configuration Loading:** Load settings (paths, models, prompts, cache config) from `config.yaml`.
2.  **Initialization:**
    - Load environment variables (API keys using `python-dotenv`).
    - Initialize LLM client (LangChain/OpenAI) and embedding model (sentence-transformers) based on `config.yaml`.
    - Load system prompt (`system_message.txt`) and explanation prompt (`explanation_prompt.txt`) from files specified in `config.yaml`.
3.  **User CV Processing:**
    - Read the user's CV (Markdown) from the path specified in `config.yaml`.
    - Generate and store the embedding vector for the user's CV.
4.  **JD Processing Loop:** For each JD found in the directory specified in `config.yaml`:
    a.  **Load JD:** Read the JD text.
    b.  **Cache Check (if enabled):**
        - Generate a cache key based on JD path, content, and relevant config (LLM model, embedding model, system prompt).
        - Attempt to load the pre-computed "Ideal CV" text and its embedding from the cache directory specified in `config.yaml`.
    c.  **Ideal CV Generation & Embedding (if cache miss):**
        - Construct the prompt using the loaded system prompt and JD text.
        - Send the prompt to the LLM (model specified in `config.yaml`) to generate an "Ideal CV".
        - Generate the embedding vector for the "Ideal CV" (model specified in `config.yaml`).
        - **Save to Cache (if enabled):** Store the generated "Ideal CV" text (`.md`) and embedding (`.npy`) in the cache directory using the generated cache key.
    d.  **Similarity Calculation:** Compute the cosine similarity between the user's CV embedding and the (cached or newly generated) "Ideal CV" embedding using `scikit-learn`.
    e.  **Conditional Explanation Generation & Caching:**
        - Check if `similarity_score >= explanation_threshold` (from `config.yaml`).
        - **If Yes:**
            - Generate explanation cache key (based on ideal CV cache key, user CV text, explanation prompt, LLM config).
            - Attempt to load explanation text from cache (`_explanation.txt`).
            - **If Cache Miss:**
                - Construct the prompt using the loaded explanation prompt, user CV text, and ideal CV text.
                - Send the prompt to the LLM to generate an explanation.
                - **Save to Cache (if enabled & successful):** Store the generated explanation text in the cache directory using the explanation cache key.
            - Store the loaded/generated explanation.
        - **If No:** Store a "skipped" message indicating the score was below the threshold.
    f.  **Storage:** Store the JD identifier, similarity score, and the explanation (or skipped message).
5.  **Ranking:** Sort the stored results by similarity score in descending order.
6.  **Output:** Print the ranked list to the console, including the JD identifier, score, and explanation for each entry.

**2. Key Technical Decisions:**
- **Language:** Python 3.x
- **Project Structure:** `src` layout.
- **Dependency Management:** `uv`.
- **Task Runner:** `just` (`justfile` defines recipes for common tasks).
- **Configuration:** Managed via `config.yaml` (loaded using `PyYAML`). Includes paths, models, prompts, cache settings, and `explanation_threshold`. API keys still loaded from `.env`.
- **LLM Interaction:** `LangChain` framework (`ChatOpenAI`, `ChatPromptTemplate`, `PromptTemplate`, `StrOutputParser`). Model configurable via `config.yaml`. System prompt refined to exclude placeholders.
- **Text Embeddings:** `sentence-transformers` library. Model configurable via `config.yaml`.
- **Vector Similarity:** `scikit-learn` (`cosine_similarity`).
- **Data Format:** Markdown (`.md`) for CV and JDs.
- **Caching:**
    - File-based caching for ideal CVs (`.md`) and embeddings (`.npy`).
    - Ideal CV cache key based on JD path, content, system prompt, LLM/embedding models.
    - **File-based caching for explanations (`.txt`).**
    - **Explanation cache key based on ideal CV key, user CV text, explanation prompt, LLM config.**
    - Cache stored in subdirectories based on the first 2 chars of the hash key.
    - Enabled/disabled, directory, and `explanation_threshold` configured via `config.yaml`.
- **Explainability:** LLM-generated explanation comparing user CV and ideal CV. **Generated and cached only if similarity score meets `explanation_threshold`.**
- **Type Checking:** `mypy` (configured in `justfile`).
- **Linting/Formatting:** `ruff` (configured in `justfile`).

**3. Core Components (within `src/rank_jobs.py`):**
- **`load_config`:** Loads and validates `config.yaml` (including `explanation_threshold`).
- **`load_api_key`:** Loads API key from `.env`.
- **`initialize_models`:** Initializes LLM and embedding models based on config.
- **Ideal CV Caching Functions (`get_cache_key`, `get_cache_paths`, `load_from_cache`, `save_to_cache`):** Manage ideal CV/embedding cache operations.
- **Explanation Caching Functions (`get_explanation_cache_key`, `get_explanation_cache_path`, `load_explanation_from_cache`, `save_explanation_to_cache`):** Manage explanation cache operations.
- **`load_text`:** Loads text from files (used for CV, JDs, prompts, cached CVs/explanations).
- **`generate_ideal_cv`:** Generates ideal CV using LLM and refined system prompt.
- **`get_embedding`:** Generates text embeddings.
- **`generate_explanation`:** Generates match explanation using LLM and explanation prompt.
- **`calculate_similarity`:** Calculates cosine similarity.
- **`main` (within `src/rank_jobs.py`):** Orchestrates the overall workflow, including loading prompts, handling caching (ideal CVs/embeddings and explanations), conditional explanation generation based on threshold, processing JDs, and printing results. This script is the primary entry point, typically run via `just rank`.

**4. Data Flow (Execution via `just rank` or `python src/rank_jobs.py`):**
```mermaid
graph TD
    subgraph Input
        Config[config.yaml]
        Env[.env]
        SysPrompt[prompts/system_message.txt]
        ExpPrompt[prompts/explanation_prompt.txt]
        UserCV[User CV (.md)]
        JD_Dir[JD Directory/*.md]
    end

    subgraph Initialization
        LoadConfig[Load config.yaml]
        LoadEnv[Load .env] --> APIKey{API Key}
        LoadPrompts[Load Prompts]
        InitModels[Initialize Models]

        LoadConfig --> InitModels
        APIKey --> InitModels
    end

    subgraph Processing
        EmbedUserCV[Embed User CV]

        LoopJDs{For Each JD} --> LoadJD[Load JD Text]

        LoadJD --> GenCacheKey[Generate Cache Key]
        GenCacheKey --> CacheCheck{Cache Hit?}

        CacheCheck -- Yes --> LoadCache[Load Ideal CV & Embedding from Cache]
        CacheCheck -- No --> GenerateIdealCV[Generate Ideal CV (LLM)]
        GenerateIdealCV --> EmbedIdealCV[Embed Ideal CV]
        EmbedIdealCV --> SaveCache[Save Ideal CV & Embedding to Cache]
        SaveCache --> JoinCache[Join]
        LoadCache --> JoinCache

        JoinCache --> CalcSim[Calculate Similarity]
        EmbedUserCV --> CalcSim

        CalcSim --> CheckThreshold{Score >= Threshold?}

        CheckThreshold -- Yes --> GenExpCacheKey[Generate Explanation Cache Key]
        UserCV -- Text --> GenExpCacheKey
        JoinCache -- Ideal CV Key --> GenExpCacheKey
        ExpPrompt -- Text --> GenExpCacheKey
        Config -- LLM Settings --> GenExpCacheKey

        GenExpCacheKey --> ExpCacheCheck{Explanation Cache Hit?}
        ExpCacheCheck -- Yes --> LoadExpCache[Load Explanation from Cache (.txt)]
        ExpCacheCheck -- No --> GenExplainLLM[Generate Explanation (LLM)]
        GenExplainLLM --> SaveExpCache[Save Explanation to Cache (.txt)]
        SaveExpCache --> JoinExp[Join Explanation]
        LoadExpCache --> JoinExp

        CheckThreshold -- No --> SkipExp[Explanation = Skipped Message]
        SkipExp --> JoinExp

        JoinExp --> StoreResult[Store JD, Score, Explanation]
        StoreResult --> LoopJDs
    end

    subgraph Output
        Rank[Rank JDs by Score]
        Print[Print Ranked List w/ Explanations]
    end

    Input --> Initialization
    Initialization --> Processing
    Processing --> Output
```

**5. Error Handling:**
- Checks for file existence (`config.yaml`, prompts, CV, JDs).
- Validates `config.yaml` structure and required keys.
- Handles `yaml.YAMLError` during config parsing.
- Handles potential API errors during LLM calls (Ideal CV & Explanation).
- Handles potential errors during embedding generation.
- Handles potential errors during cache loading/saving.
- Gracefully skips processing a JD if critical errors occur (e.g., file load, ideal CV generation, embedding).
- Allows continuation without explanations if the explanation prompt fails to load.
