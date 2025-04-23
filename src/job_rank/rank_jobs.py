# Main script for Job Description Ranker

import glob
import hashlib  # Added for caching
import os
from pathlib import Path

import numpy as np  # Added for saving/loading embeddings
import yaml
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate  # Added PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

# --- Configuration Loading ---


def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    path = Path(config_path)
    """Loads configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        # Basic validation
        required_keys = ["cv_path", "jd_dir", "llm", "embedding", "prompts", "cache"]
        if not all(k in config for k in required_keys):
            raise ValueError(
                f"Config file is missing required top-level keys: {required_keys}"
            )
        if not all(k in config["llm"] for k in ["model_name", "temperature"]):
            raise ValueError(
                "Config file is missing required 'llm' keys: ['model_name', 'temperature']"
            )
        if "model_name" not in config["embedding"]:
            raise ValueError(
                "Config file is missing required 'embedding' key: 'model_name'"
            )
        # Check for both prompt files now
        if not all(
            k in config["prompts"]
            for k in ["system_message_file", "explanation_prompt_file"]
        ):
            raise ValueError(
                "Config file is missing required 'prompts' keys: ['system_message_file', 'explanation_prompt_file']"
            )
        if not all(
            k in config["cache"]
            for k in ["enabled", "directory", "explanation_threshold"]
        ):
            raise ValueError(
                "Config file is missing required 'cache' keys: ['enabled', 'directory', 'explanation_threshold']"
            )
        # Validate threshold type
        if not isinstance(config["cache"]["explanation_threshold"], (int, float)):
            raise ValueError("'explanation_threshold' in config must be a number.")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file: {e}")
    # Let other exceptions (like our explicit ValueErrors for missing keys) propagate


# --- Initialization ---


def load_api_key():
    """Loads the OpenAI API key from .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    return api_key


def initialize_models(api_key, config):
    """Initializes the LLM and embedding models based on config."""
    llm_config = config["llm"]
    embedding_config = config["embedding"]
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model=llm_config["model_name"],
        temperature=llm_config["temperature"],
    )
    embedding_model = SentenceTransformer(embedding_config["model_name"])
    return llm, embedding_model


# --- Caching Functions ---


def get_cache_key(jd_path, jd_text, config, system_prompt_text):
    """Generates a unique cache key based on JD content and relevant config."""
    hasher = hashlib.sha256()
    hasher.update(jd_path.encode("utf-8"))
    hasher.update(jd_text.encode("utf-8"))
    hasher.update(config["llm"]["model_name"].encode("utf-8"))
    hasher.update(str(config["llm"]["temperature"]).encode("utf-8"))
    hasher.update(config["embedding"]["model_name"].encode("utf-8"))
    hasher.update(system_prompt_text.encode("utf-8"))  # Include prompt in hash
    return hasher.hexdigest()


def get_cache_paths(cache_dir, cache_key):
    """Gets the file paths for cached ideal CV and embedding."""
    cache_subdir = Path(cache_dir) / cache_key[:2]  # Use first 2 chars for subdir
    ideal_cv_path = cache_subdir / f"{cache_key}_ideal_cv.md"
    embedding_path = cache_subdir / f"{cache_key}_embedding.npy"
    return ideal_cv_path, embedding_path


def load_from_cache(cache_dir, cache_key):
    """Loads ideal CV text and embedding from cache if available."""
    ideal_cv_path, embedding_path = get_cache_paths(cache_dir, cache_key)
    if ideal_cv_path.is_file() and embedding_path.is_file():
        try:
            print(f"Cache hit for key {cache_key[:8]}...")
            ideal_cv_text = load_text(ideal_cv_path)
            embedding = np.load(embedding_path)
            if ideal_cv_text is not None and embedding is not None:
                return ideal_cv_text, embedding
        except Exception as e:
            print(f"Error loading from cache for key {cache_key[:8]}: {e}")
    return None, None  # Cache miss or error


def save_to_cache(cache_dir, cache_key, ideal_cv_text, ideal_cv_embedding):
    """Saves ideal CV text and embedding to the cache."""
    ideal_cv_path, embedding_path = get_cache_paths(cache_dir, cache_key)
    try:
        print(f"Saving to cache for key {cache_key[:8]}...")
        # Ensure cache directory exists
        ideal_cv_path.parent.mkdir(parents=True, exist_ok=True)
        # Save ideal CV text
        with open(ideal_cv_path, "w", encoding="utf-8") as f:
            f.write(ideal_cv_text)
        # Save embedding
        np.save(embedding_path, ideal_cv_embedding)
        print(f"Successfully saved to cache: {ideal_cv_path.parent}")
    except Exception as e:
        print(f"Error saving to cache for key {cache_key[:8]}: {e}")


# --- Explanation Caching Functions ---


def get_explanation_cache_key(
    ideal_cv_cache_key, user_cv_text, explanation_prompt_text, config
):
    """Generates a unique cache key for the explanation."""
    hasher = hashlib.sha256()
    # Base key depends on the ideal CV (which depends on JD, system prompt, models)
    hasher.update(ideal_cv_cache_key.encode("utf-8"))
    # Also depends on the specific user CV
    hasher.update(user_cv_text.encode("utf-8"))
    # And the explanation prompt used
    hasher.update(explanation_prompt_text.encode("utf-8"))
    # And the LLM model used for explanation (same as ideal CV for now)
    hasher.update(config["llm"]["model_name"].encode("utf-8"))
    hasher.update(str(config["llm"]["temperature"]).encode("utf-8"))
    return hasher.hexdigest()


def get_explanation_cache_path(cache_dir, explanation_cache_key):
    """Gets the file path for a cached explanation."""
    cache_subdir = Path(cache_dir) / explanation_cache_key[:2]
    return cache_subdir / f"{explanation_cache_key}_explanation.txt"


def load_explanation_from_cache(cache_dir, explanation_cache_key):
    """Loads explanation text from cache if available."""
    explanation_path = get_explanation_cache_path(cache_dir, explanation_cache_key)
    if explanation_path.is_file():
        try:
            print(f"Explanation cache hit for key {explanation_cache_key[:8]}...")
            explanation_text = load_text(explanation_path)
            if explanation_text is not None:
                return explanation_text
        except Exception as e:
            print(
                f"Error loading explanation from cache for key {explanation_cache_key[:8]}: {e}"
            )
    return None  # Cache miss or error


def save_explanation_to_cache(cache_dir, explanation_cache_key, explanation_text):
    """Saves explanation text to the cache."""
    explanation_path = get_explanation_cache_path(cache_dir, explanation_cache_key)
    try:
        print(f"Saving explanation to cache for key {explanation_cache_key[:8]}...")
        explanation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(explanation_path, "w", encoding="utf-8") as f:
            f.write(explanation_text)
        print(f"Successfully saved explanation to cache: {explanation_path.parent}")
    except Exception as e:
        print(
            f"Error saving explanation to cache for key {explanation_cache_key[:8]}: {e}"
        )


# --- Core Functions ---


def load_text(filepath):
    """Loads text content from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def generate_ideal_cv(llm, job_description, system_prompt_text):
    """Generates an 'Ideal CV' based on the job description using the LLM and loaded system prompt."""
    print("Generating Ideal CV for JD...")  # Add JD identifier if possible later
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            (
                "human",
                "Job Description:\n\n{job_description}",
            ),  # Keep human template simple for now
        ]
    )
    chain = prompt_template | llm | StrOutputParser()
    try:
        ideal_cv = chain.invoke({"job_description": job_description})
    except Exception as e:
        print(f"Error during LLM call for Ideal CV generation: {e}")
        return None
    return ideal_cv


def get_embedding(model, text):
    """Generates embedding for the given text."""
    print("Generating embedding...")
    try:
        embedding = model.encode(text)
        # Reshape for cosine_similarity which expects 2D arrays
        return embedding.reshape(1, -1)
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None


def generate_explanation(llm, explanation_prompt_text, user_cv, ideal_cv):
    """Generates an explanation for the match between user CV and ideal CV."""
    print("Generating explanation...")
    # Use a simple prompt template for explanation
    prompt = PromptTemplate.from_template(explanation_prompt_text)
    chain = prompt | llm | StrOutputParser()
    try:
        explanation = chain.invoke({"user_cv": user_cv, "ideal_cv": ideal_cv})
        return explanation.strip()
    except Exception as e:
        print(f"Error during LLM call for explanation generation: {e}")
        return "Explanation generation failed."


def calculate_similarity(embedding1, embedding2):
    """Calculates cosine similarity between two embeddings."""
    # print("Calculating similarity...") # Less verbose
    # Ensure both are 2D arrays
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


# --- Core Pipeline ---


def run_ranking_pipeline(config_path="config.yaml") -> list[dict]:
    """Runs the full job ranking pipeline and returns the results."""
    print("Starting Job Description Ranking Pipeline...")
    # --- Load Config, API Key, Models, Prompts ---
    config = load_config(config_path)
    print("Configuration loaded.")

    api_key = load_api_key()
    llm, embedding_model = initialize_models(api_key, config)
    print("Models initialized.")

    # Load system prompt
    system_prompt_path = config["prompts"]["system_message_file"]
    print(f"Loading system prompt from: {system_prompt_path}")
    system_prompt_text = load_text(system_prompt_path)
    if not system_prompt_text:
        raise FileNotFoundError(f"System prompt not found at {system_prompt_path}")
    print("System prompt loaded.")

    # Load explanation prompt
    explanation_prompt_path = config["prompts"]["explanation_prompt_file"]
    print(f"Loading explanation prompt from: {explanation_prompt_path}")
    explanation_prompt_text = load_text(explanation_prompt_path)
    if not explanation_prompt_text:
        print(
            "Warning: Failed to load explanation prompt. Explanations will be skipped."
        )
        explanation_prompt_text = None  # Allow continuation

    cache_config = config["cache"]
    cache_enabled = cache_config["enabled"]
    cache_dir = Path(cache_config["directory"]) if cache_enabled else None
    if cache_enabled:
        print(f"Caching enabled. Cache directory: {cache_dir}")

    # --- Load and Embed User CV ---
    cv_path = config["cv_path"]
    print(f"Loading user CV from: {cv_path}")
    user_cv_text = load_text(cv_path)
    if not user_cv_text:
        raise FileNotFoundError(f"User CV not found at {cv_path}")

    print("Embedding user CV...")
    user_cv_embedding = get_embedding(embedding_model, user_cv_text)
    if user_cv_embedding is None:
        raise ValueError("Failed to embed user CV.")
    print("User CV embedded.")

    # --- Process Job Descriptions ---
    results = []
    jd_dir = config["jd_dir"]
    jd_files = glob.glob(os.path.join(jd_dir, "*.md"))
    print(f"Found {len(jd_files)} job descriptions in {jd_dir}")

    if not jd_files:
        print(f"No job description files (.md) found in {jd_dir}.")
        return []  # Return empty list

    for jd_path in jd_files:
        jd_filename = os.path.basename(jd_path)
        print(f"\nProcessing JD: {jd_filename}")

        # Load JD
        jd_text = load_text(jd_path)
        if not jd_text:
            print(f"Skipping {jd_filename} due to loading error.")
            continue

        ideal_cv_text = None
        ideal_cv_embedding = None
        cache_key = None

        # --- Cache Check ---
        if cache_enabled:
            cache_key = get_cache_key(jd_path, jd_text, config, system_prompt_text)
            ideal_cv_text, ideal_cv_embedding = load_from_cache(cache_dir, cache_key)

        # --- Generation/Embedding (if not cached) ---
        if ideal_cv_text is None or ideal_cv_embedding is None:
            if cache_enabled and cache_key:  # Check cache_key exists before printing
                print(f"Cache miss for key {cache_key[:8]}...")

            # Generate Ideal CV
            ideal_cv_text = generate_ideal_cv(llm, jd_text, system_prompt_text)
            if not ideal_cv_text:
                print(f"Failed to generate Ideal CV for {jd_filename}. Skipping.")
                continue

            # Embed Ideal CV
            ideal_cv_embedding = get_embedding(embedding_model, ideal_cv_text)
            if ideal_cv_embedding is None:
                print(f"Failed to embed Ideal CV for {jd_filename}. Skipping.")
                continue

            # --- Save to Cache ---
            if cache_enabled and cache_key:
                save_to_cache(cache_dir, cache_key, ideal_cv_text, ideal_cv_embedding)

        # --- Calculate Similarity ---
        similarity_score = calculate_similarity(user_cv_embedding, ideal_cv_embedding)
        print(f"Similarity score for {jd_filename}: {similarity_score:.4f}")

        # --- Generate/Cache Explanation (Conditional) ---
        explanation = None  # Default to None
        explanation_threshold = config["cache"]["explanation_threshold"]

        if not explanation_prompt_text:
            explanation = "Explanation skipped (prompt not loaded)."
        elif not ideal_cv_text:
            explanation = "Explanation skipped (ideal CV generation failed)."
        elif similarity_score >= explanation_threshold:
            # Only proceed if score is above threshold
            explanation_cache_key = None
            if cache_enabled and cache_key:  # Need ideal_cv cache key
                explanation_cache_key = get_explanation_cache_key(
                    cache_key, user_cv_text, explanation_prompt_text, config
                )
                explanation = load_explanation_from_cache(
                    cache_dir, explanation_cache_key
                )

            if explanation is None:  # Cache miss or caching disabled
                if cache_enabled and explanation_cache_key:  # Check key exists
                    print(
                        f"Explanation cache miss for key {explanation_cache_key[:8]}..."
                    )
                explanation = generate_explanation(
                    llm, explanation_prompt_text, user_cv_text, ideal_cv_text
                )
                # Save to cache if generated successfully and caching enabled
                if (
                    explanation != "Explanation generation failed."
                    and cache_enabled
                    and explanation_cache_key
                ):
                    save_explanation_to_cache(
                        cache_dir, explanation_cache_key, explanation
                    )
        else:
            # Score is below threshold
            explanation = f"Explanation skipped (score {similarity_score:.4f} < threshold {explanation_threshold})."

        results.append(
            {
                "jd": jd_filename,
                "score": similarity_score,
                "explanation": explanation,
            }
        )

    # --- Rank Results ---
    print("\nRanking complete.")
    results.sort(key=lambda x: x["score"], reverse=True)
    return results  # Return the data structure


# --- Main Execution (CLI Entry Point) ---


def main():
    """CLI entry point to run the pipeline and print results."""
    print("Starting Job Description Ranker (CLI)...")
    try:
        # Run the core pipeline
        ranked_results = run_ranking_pipeline()  # Use default config path

        # Print the results
        if not ranked_results:
            print("No job descriptions were successfully processed.")
            return

        print("\n--- Final Ranked Results ---")
        for i, result in enumerate(ranked_results):
            print(f"\n--- Rank {i + 1} ---")
            print(f"Job Description: {result['jd']}")
            print(f"Similarity Score: {result['score']:.4f}")
            print(f"Explanation:\n{result['explanation']}")
            print("-" * 20)

    except FileNotFoundError as fnfe:
        print(f"Error: Required file not found: {fnfe}")
    except ValueError as ve:
        print(f"Configuration or Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
