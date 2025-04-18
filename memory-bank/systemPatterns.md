# System Patterns: Job Description Ranker

**1. Overall Architecture:**
Command-Line Application (Python Script)

**Workflow:**
1.  **Initialization:** Load environment variables (API keys), initialize LLM client (LangChain/OpenAI) and embedding model (sentence-transformers).
2.  **Input:** Read the user's CV (Markdown) and all JDs (Markdown) from specified locations.
3.  **CV Embedding:** Generate the embedding vector for the user's CV.
4.  **JD Processing Loop:** For each JD:
    a.  **Ideal CV Generation:** Construct a prompt using the JD text. Send the prompt to the LLM (GPT-4o-mini via LangChain) to generate an "Ideal CV".
    b.  **Ideal CV Embedding:** Generate the embedding vector for the generated "Ideal CV".
    c.  **Similarity Calculation:** Compute the cosine similarity between the user's CV embedding and the "Ideal CV" embedding using `scikit-learn`.
    d.  **Storage:** Store the JD identifier (e.g., filename) and its similarity score.
5.  **Ranking:** Sort the stored results by similarity score in descending order.
6.  **Output:** Print the ranked list to the console.

**2. Key Technical Decisions (Initial):**
- **Language:** Python 3.x
- **Dependency Management:** `uv`
- **LLM Interaction:** `LangChain` framework with `langchain-openai` integration.
    - Model: `gpt-4o-mini` (initially).
- **Text Embeddings:** `sentence-transformers` library.
    - Model: `all-MiniLM-L6-v2` (initially).
- **Vector Similarity:** `scikit-learn` (`cosine_similarity`).
- **Data Format:** Markdown (`.md`) for CV and JDs.
- **Configuration:** API keys loaded from `.env` file using `python-dotenv`.

**3. Core Components:**
- **`rank_jobs.py`:** Main script orchestrating the workflow. Contains functions for loading data, initializing models, generating ideal CVs, embedding text, calculating similarity, and the main execution logic.
- **LLM Interaction Module (within `rank_jobs.py`):** Handles prompt creation and communication with the OpenAI API via LangChain (`generate_ideal_cv` function).
- **Embedding Module (within `rank_jobs.py`):** Loads the `sentence-transformers` model and encodes text (`get_embedding` function).
- **Similarity Module:** Calculates cosine similarity.
- **Data Loading Module:** Reads Markdown files.

**4. Data Flow:**
```mermaid
graph TD
    subgraph Input
        CV[CV.md]
        JD_Dir[JD Directory/*.md]
    end

    subgraph Processing
        LoadData[Load Data] --> EmbedCV[Embed User CV]
        LoadData --> LoopJDs{For Each JD}

        LoopJDs --> GenerateIdealCV[Generate Ideal CV (LLM)]
        GenerateIdealCV --> EmbedIdealCV[Embed Ideal CV]
        EmbedCV --> CalcSim[Calculate Similarity]
        EmbedIdealCV --> CalcSim

        CalcSim --> StoreScore[Store JD + Score]
        StoreScore --> LoopJDs
    end

    subgraph Output
        Rank[Rank JDs by Score]
        Print[Print Ranked List]
    end

    Input --> Processing
    Processing --> Output
```

**5. Error Handling (Minimal MVP):**
- Basic checks for file existence.
- Catch potential API errors during LLM calls.
- Assume successful embedding generation.
