# Technical Context: Job Description Ranker

**1. Core Technologies:**
- **Python:** Version 3.x (as defined by `.python-version` if present, otherwise assume a recent version like 3.10+).
- **`uv`:** For Python package management and virtual environments.
- **`LangChain`:** Framework for orchestrating LLM interactions.
- **`langchain-openai`:** LangChain integration for OpenAI models.
- **`openai`:** Underlying client library for OpenAI API (likely pulled in by `langchain-openai`).
- **`sentence-transformers`:** Library for generating text embeddings.
- **`scikit-learn`:** For calculating cosine similarity and potentially other ML utilities.
- **`python-dotenv`:** For loading environment variables (like API keys) from a `.env` file.
- **`markdown`:** Python library for parsing Markdown files (useful if preprocessing/cleaning is needed before embedding or sending to LLM).

**2. Development Environment:**
- **OS:** Linux (based on environment details).
- **IDE:** VS Code with relevant extensions.
- **Containerization:** Running within a Dev Container (implied by `.devcontainer` files).
- **Version Control:** Git (assumed, `.gitignore` exists).

**3. External Services & APIs:**
- **OpenAI API:** Required for LLM generation (using `gpt-4o-mini`). Requires an `OPENAI_API_KEY`.

**4. Key Models:**
- **LLM:** `gpt-4o-mini` (via OpenAI API).
- **Embedding Model:** `all-MiniLM-L6-v2` (via `sentence-transformers`). This model is downloaded locally by the library on first use.

**5. Setup & Installation:**
- Ensure Python and `uv` are installed.
- Create a virtual environment (managed by `uv`).
- Install dependencies using `uv add langchain langchain-openai sentence-transformers scikit-learn python-dotenv markdown`.
- Create a `.env` file in the project root containing `OPENAI_API_KEY=your_actual_key`. Ensure `.env` is added to `.gitignore`.

**6. Technical Constraints & Considerations:**
- **API Costs:** Calls to the OpenAI API incur costs. `gpt-4o-mini` is relatively cost-effective, but usage should be monitored.
- **API Rate Limits:** OpenAI API has rate limits. For batch processing many JDs, potential delays or error handling might be needed.
- **Embedding Model Size:** `all-MiniLM-L6-v2` is relatively small, but requires download space (~90MB).
- **Processing Time:** LLM generation is the slowest step. Embedding is generally faster but depends on text length and hardware. Cosine similarity is very fast.
- **Markdown Parsing:** Assumes reasonably clean Markdown. Complex or malformed Markdown might require more robust parsing.
