# Main script for Job Description Ranker

import os
import glob
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Constants ---
CV_PATH = "data/cv/sample_cv.md"
JD_DIR = "data/jd"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = "gpt-4o-mini"

# --- Initialization ---

def load_api_key():
    """Loads the OpenAI API key from .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    return api_key

def initialize_models(api_key):
    """Initializes the LLM and embedding models."""
    llm = ChatOpenAI(openai_api_key=api_key, model=LLM_MODEL, temperature=0.2)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return llm, embedding_model

# --- Core Functions (Placeholders) ---

def load_text(filepath):
    """Loads text content from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def generate_ideal_cv(llm, job_description):
    """Generates an 'Ideal CV' based on the job description using the LLM."""
    # Placeholder - Implementation needed
    print("Generating Ideal CV for JD...") # Add JD identifier if possible later
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert CV writer. Based *only* on the provided job description, create a concise, ideal CV highlighting the key skills, experience, and qualifications required. Focus on extracting and structuring the requirements from the JD into a CV format. Do not add information not present in the JD."),
            ("human", "Job Description:\n\n{job_description}"),
        ]
    )
    chain = prompt_template | llm | StrOutputParser()
    ideal_cv = chain.invoke({"job_description": job_description})
    return ideal_cv

def get_embedding(model, text):
    """Generates embedding for the given text."""
    # Placeholder - Implementation needed
    print("Generating embedding...")
    embedding = model.encode(text)
    # Reshape for cosine_similarity which expects 2D arrays
    return embedding.reshape(1, -1)

def calculate_similarity(embedding1, embedding2):
    """Calculates cosine similarity between two embeddings."""
    # Placeholder - Implementation needed
    print("Calculating similarity...")
    # Ensure both are 2D arrays
    if embedding1.ndim == 1: embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1: embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

# --- Main Execution ---

def main():
    print("Starting Job Description Ranker...")
    try:
        api_key = load_api_key()
        llm, embedding_model = initialize_models(api_key)
        print("Models initialized.")

        # 1. Load and embed the user's CV
        print(f"Loading user CV from: {CV_PATH}")
        user_cv_text = load_text(CV_PATH)
        if not user_cv_text:
            print("Failed to load user CV. Exiting.")
            return

        print("Embedding user CV...")
        user_cv_embedding = get_embedding(embedding_model, user_cv_text)
        print("User CV embedded.")

        # 2. Process Job Descriptions
        results = []
        jd_files = glob.glob(os.path.join(JD_DIR, "*.md"))
        print(f"Found {len(jd_files)} job descriptions in {JD_DIR}")

        if not jd_files:
            print("No job description files (.md) found. Exiting.")
            return

        for jd_path in jd_files:
            jd_filename = os.path.basename(jd_path)
            print(f"\nProcessing JD: {jd_filename}")

            # Load JD
            jd_text = load_text(jd_path)
            if not jd_text:
                print(f"Skipping {jd_filename} due to loading error.")
                continue

            # Generate Ideal CV
            ideal_cv_text = generate_ideal_cv(llm, jd_text)
            if not ideal_cv_text:
                 print(f"Failed to generate Ideal CV for {jd_filename}. Skipping.")
                 continue
            # print(f"--- Ideal CV for {jd_filename} ---\n{ideal_cv_text}\n-----------------------------") # Optional: print generated CV

            # Embed Ideal CV
            ideal_cv_embedding = get_embedding(embedding_model, ideal_cv_text)

            # Calculate Similarity
            similarity_score = calculate_similarity(user_cv_embedding, ideal_cv_embedding)
            print(f"Similarity score for {jd_filename}: {similarity_score:.4f}")

            results.append({"jd": jd_filename, "score": similarity_score})

        # 3. Rank and Print Results
        if not results:
            print("No job descriptions were successfully processed.")
            return

        print("\n--- Ranking Results ---")
        results.sort(key=lambda x: x["score"], reverse=True)

        for i, result in enumerate(results):
            print(f"{i+1}. {result['jd']} - Score: {result['score']:.4f}")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
