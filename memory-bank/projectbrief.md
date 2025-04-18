# Project Brief: Job Description Ranker

**1. Project Goal:**
Develop a system to rank job descriptions (JDs) based on their relevance to a given Curriculum Vitae (CV).

**2. Core Problem:**
Manually comparing a CV against numerous JDs is time-consuming and subjective. Existing keyword-based methods lack nuanced understanding of skills and experience alignment.

**3. Proposed Solution:**
Instead of direct CV-JD comparison, the system will:
    a. Use a Large Language Model (LLM) to generate an "Ideal CV" based on the requirements outlined in each JD.
    b. Convert the original CV and the generated "Ideal CV" into numerical vector embeddings.
    c. Calculate the cosine similarity between the original CV's embedding and the "Ideal CV's" embedding.
    d. Rank the JDs based on this similarity score, with higher scores indicating better relevance.

**4. Key Features:**
    - Input: One CV (Markdown format) and multiple JDs (Markdown format).
    - Processing: LLM-based ideal CV generation, text embedding, cosine similarity calculation.
    - Output: A ranked list of JDs (identifiers and scores) printed to the console.

**5. Initial Scope (MVP):**
    - Use synthetic/placeholder data for CV and JDs.
    - Implement the core ranking logic using LangChain, OpenAI (GPT-4o-mini), sentence-transformers, and scikit-learn.
    - Focus on demonstrating the workflow, not optimizing performance or handling complex edge cases.
    - Manage dependencies with `uv`.
    - Load API keys from a `.env` file.

**6. Success Criteria:**
    - The system successfully processes a sample CV and JDs.
    - It generates ideal CVs using the LLM.
    - It calculates similarity scores and outputs a ranked list.
    - The ranking appears logical based on the synthetic data (manual check).
