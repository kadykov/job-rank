# System Patterns: Job Description Ranker

**Date:** 2025-04-20

**1. Overall Architecture:**
Command-Line Application (Python Script) using a `src` layout.

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
    e.  **Explanation Generation:**
        - Construct the prompt using the loaded explanation prompt, the user's CV text, and the (cached or newly generated) "Ideal CV" text.
        - Send the prompt to the LLM to generate an explanation of the match.
    f.  **Storage:** Store the JD identifier, similarity score, and explanation.
5.  **Ranking:** Sort the stored results by similarity score in descending order.
6.  **Output:** Print the ranked list to the console, including the JD identifier, score, and explanation for each entry.

**2. Key Technical Decisions:**
- **Language:** Python 3.x
- **Project Structure:** `src` layout.
- **Dependency Management:** `uv`.
- **Configuration:** Managed via `config.yaml` (loaded using `PyYAML`). API keys still loaded from `.env`.
- **LLM Interaction:** `LangChain` framework (`ChatOpenAI`, `ChatPromptTemplate`, `PromptTemplate`, `StrOutputParser`). Model configurable via `config.yaml`.
- **Text Embeddings:** `sentence-transformers` library. Model configurable via `config.yaml`.
- **Vector Similarity:** `scikit-learn` (`cosine_similarity`).
- **Data Format:** Markdown (`.md`) for CV and JDs.
- **Caching:**
    - File-based caching for ideal CVs (`.md`) and embeddings (`.npy`).
    - SHA256 hash of JD path, content, and relevant config used as cache key.
    - Cache stored in subdirectories based on the first 2 chars of the hash key.
    - Enabled/disabled and directory configured via `config.yaml`.
- **Explainability:** LLM-generated explanation comparing user CV and ideal CV.

**3. Core Components (within `src/rank_jobs.py`):**
- **`load_config`:** Loads and validates `config.yaml`.
- **`load_api_key`:** Loads API key from `.env`.
- **`initialize_models`:** Initializes LLM and embedding models based on config.
- **Caching Functions (`get_cache_key`, `get_cache_paths`, `load_from_cache`, `save_to_cache`):** Manage cache operations.
- **`load_text`:** Loads text from files (used for CV, JDs, prompts, cached CVs).
- **`generate_ideal_cv`:** Generates ideal CV using LLM and system prompt.
- **`get_embedding`:** Generates text embeddings.
- **`generate_explanation`:** Generates match explanation using LLM and explanation prompt.
- **`calculate_similarity`:** Calculates cosine similarity.
- **`main`:** Orchestrates the overall workflow, including loading prompts, handling caching, processing JDs, and printing results.

**4. Data Flow:**
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

        JoinCache --> GenExplain[Generate Explanation (LLM)]
        UserCV -- Text --> GenExplain

        CalcSim --> StoreResult[Store JD, Score, Explanation]
        GenExplain --> StoreResult
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
