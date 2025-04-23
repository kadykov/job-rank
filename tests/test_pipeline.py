from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# Project should be installed editable, so import directly from the package name
from rank_jobs import run_ranking_pipeline

# --- Test Fixtures ---


@pytest.fixture
def temp_test_env(tmp_path):
    """Creates a temporary directory structure with config, prompts, CV, and JDs."""
    base_path = tmp_path
    config_path = base_path / "test_config.yaml"
    prompts_dir = base_path / "prompts"
    cv_dir = base_path / "cv"
    jd_dir = base_path / "jd"
    cache_dir = base_path / "cache"

    # Create directories
    prompts_dir.mkdir()
    cv_dir.mkdir()
    jd_dir.mkdir()
    # cache_dir will be created by the function if needed

    # Create dummy files
    (prompts_dir / "system.txt").write_text("System Prompt Content")
    (prompts_dir / "explain.txt").write_text("Explain Prompt: {user_cv} vs {ideal_cv}")
    (cv_dir / "my_cv.md").write_text("My CV Content")
    (jd_dir / "jd1.md").write_text("Job Description 1")
    (jd_dir / "jd2.md").write_text("Job Description 2")

    # Create config file pointing to temp files
    config_data = {
        "cv_path": str(cv_dir / "my_cv.md"),
        "jd_dir": str(jd_dir),
        "llm": {"model_name": "test-llm", "temperature": 0.5},
        "embedding": {"model_name": "test-embedding"},
        "prompts": {
            "system_message_file": str(prompts_dir / "system.txt"),
            "explanation_prompt_file": str(prompts_dir / "explain.txt"),
        },
        "cache": {
            "enabled": True,
            "directory": str(cache_dir),
            "explanation_threshold": 0.6,
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return {
        "config_path": str(config_path),
        "jd_dir": str(jd_dir),
        "jd_files": [str(jd_dir / "jd1.md"), str(jd_dir / "jd2.md")],
        "cache_dir": str(cache_dir),
    }


# --- Test Cases ---


@patch("rank_jobs.load_api_key")
@patch("rank_jobs.initialize_models")
@patch("rank_jobs.get_embedding")
@patch("rank_jobs.generate_ideal_cv")
@patch("rank_jobs.generate_explanation")
@patch("rank_jobs.glob.glob")  # Mock glob to control found files
def test_run_ranking_pipeline_success(
    mock_glob,
    mock_generate_explanation,
    mock_generate_ideal_cv,
    mock_get_embedding,
    mock_initialize_models,
    mock_load_api_key,
    temp_test_env,
):
    """Tests the successful execution of the ranking pipeline with mocks."""
    # --- Mock Configuration ---
    mock_load_api_key.return_value = "dummy_api_key"
    mock_llm = MagicMock()
    mock_embedding_model = MagicMock()
    mock_initialize_models.return_value = (mock_llm, mock_embedding_model)

    # Mock glob to return only the files created in the fixture
    mock_glob.return_value = temp_test_env["jd_files"]

    # Mock embeddings (needs to return 2D array)
    mock_user_cv_embedding = np.array([[0.1, 0.9]])
    mock_ideal_cv1_embedding = np.array([[0.2, 0.8]])  # High similarity
    mock_ideal_cv2_embedding = np.array([[0.9, 0.1]])  # Low similarity
    # Use side_effect to return different embeddings for user CV vs ideal CVs
    mock_get_embedding.side_effect = [
        mock_user_cv_embedding,  # First call for user CV
        mock_ideal_cv1_embedding,  # Second call for ideal CV 1
        mock_ideal_cv2_embedding,  # Third call for ideal CV 2
    ]

    # Mock LLM generations
    mock_generate_ideal_cv.side_effect = ["Ideal CV 1", "Ideal CV 2"]
    mock_generate_explanation.return_value = "Generated Explanation"

    # --- Execute Pipeline ---
    results = run_ranking_pipeline(config_path=temp_test_env["config_path"])

    # --- Assertions ---
    assert isinstance(results, list)
    assert len(results) == 2  # Should match the number of JDs mocked by glob

    # Check structure of results
    assert "jd" in results[0]
    assert "score" in results[0]
    assert "explanation" in results[0]
    assert "jd" in results[1]
    assert "score" in results[1]
    assert "explanation" in results[1]

    # Check sorting (JD1 should have higher similarity based on mock embeddings)
    assert results[0]["jd"] == "jd1.md"
    assert results[1]["jd"] == "jd2.md"
    assert results[0]["score"] > results[1]["score"]

    # Check scores are floats (approximate check)
    assert isinstance(results[0]["score"], float)
    assert isinstance(results[1]["score"], float)

    # Check explanation generation (JD1 score should be > 0.6, JD2 < 0.6)
    # Note: Actual cosine similarity of mocks:
    # (0.1*0.2 + 0.9*0.8) / (sqrt(0.1^2+0.9^2)*sqrt(0.2^2+0.8^2)) = 0.74 / (sqrt(0.82)*sqrt(0.68)) ~= 0.99
    # (0.1*0.9 + 0.9*0.1) / (sqrt(0.1^2+0.9^2)*sqrt(0.9^2+0.1^2)) = 0.18 / 0.82 ~= 0.22
    assert results[0]["explanation"] == "Generated Explanation"
    assert (
        "skipped (score" in results[1]["explanation"]
    )  # Check if explanation was skipped due to threshold

    # Verify mocks were called
    mock_load_api_key.assert_called_once()
    mock_initialize_models.assert_called_once()
    assert mock_get_embedding.call_count == 3  # User CV + 2 Ideal CVs
    assert (
        mock_generate_ideal_cv.call_count == 2
    )  # One for each JD (assuming cache miss)
    mock_generate_explanation.assert_called_once()  # Only called for JD1

    # Check if cache files were created (basic check)
    cache_dir = Path(temp_test_env["cache_dir"])
    assert cache_dir.exists()
    # More specific checks could verify specific cache file existence/content
    # based on calculated cache keys if needed.
