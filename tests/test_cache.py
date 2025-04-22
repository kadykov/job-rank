import pytest
import hashlib
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Functions to test (import directly from the installed module)
from rank_jobs import (
    get_cache_key,
    get_cache_paths,
    load_from_cache,
    save_to_cache,
    load_text, # Helper needed for load_from_cache mock
    # New functions for explanation caching
    get_explanation_cache_key,
    get_explanation_cache_path,
    load_explanation_from_cache,
    save_explanation_to_cache
)
# Need to patch functions within the correct module path now
MODULE_PATH = "rank_jobs"

# --- Test Data ---

MOCK_CONFIG = {
    'llm': {'model_name': 'test-llm', 'temperature': 0.5},
    'embedding': {'model_name': 'test-emb'},
    'cache': {'directory': 'dummy_cache'} # Used by get_cache_paths indirectly
}
MOCK_JD_PATH = "data/jd/test_jd.md"
MOCK_JD_TEXT = "This is a test job description."
MOCK_SYSTEM_PROMPT = "Test system prompt."
EXPECTED_HASH_INPUT = (
    MOCK_JD_PATH + MOCK_JD_TEXT +
    MOCK_CONFIG['llm']['model_name'] +  # type: ignore[index] # Suppress mypy error for dict access
    str(MOCK_CONFIG['llm']['temperature']) +  # type: ignore[index] # Suppress mypy error for dict access
    MOCK_CONFIG['embedding']['model_name'] +  # type: ignore[index] # Suppress mypy error for dict access
    MOCK_SYSTEM_PROMPT
)
EXPECTED_CACHE_KEY = hashlib.sha256(EXPECTED_HASH_INPUT.encode('utf-8')).hexdigest()
EXPECTED_CACHE_SUBDIR = Path('dummy_cache') / EXPECTED_CACHE_KEY[:2]
EXPECTED_IDEAL_CV_PATH = EXPECTED_CACHE_SUBDIR / f"{EXPECTED_CACHE_KEY}_ideal_cv.md"
EXPECTED_EMBEDDING_PATH = EXPECTED_CACHE_SUBDIR / f"{EXPECTED_CACHE_KEY}_embedding.npy"

MOCK_IDEAL_CV_TEXT = "Generated ideal CV."
MOCK_EMBEDDING = np.array([[0.1, 0.2, 0.3]])

# --- Test Data for Explanation Caching ---
MOCK_USER_CV_TEXT = "This is the user's CV."
MOCK_EXPLANATION_PROMPT_TEXT = "Explain the match."
MOCK_EXPLANATION_TEXT = "This is the explanation."
# Use existing MOCK_CONFIG and EXPECTED_CACHE_KEY (as ideal_cv_cache_key)
EXPECTED_EXPLANATION_HASH_INPUT = (
    EXPECTED_CACHE_KEY +
    MOCK_USER_CV_TEXT +
    MOCK_EXPLANATION_PROMPT_TEXT +
    MOCK_CONFIG['llm']['model_name'] + # type: ignore[index]
    str(MOCK_CONFIG['llm']['temperature']) # type: ignore[index]
)
EXPECTED_EXPLANATION_CACHE_KEY = hashlib.sha256(EXPECTED_EXPLANATION_HASH_INPUT.encode('utf-8')).hexdigest()
EXPECTED_EXPLANATION_CACHE_PATH = (
    Path('dummy_cache') /
    EXPECTED_EXPLANATION_CACHE_KEY[:2] /
    f"{EXPECTED_EXPLANATION_CACHE_KEY}_explanation.txt"
)

# --- Tests for get_cache_key ---

def test_get_cache_key_consistency():
    """Tests that the cache key is consistent for the same inputs."""
    key1 = get_cache_key(MOCK_JD_PATH, MOCK_JD_TEXT, MOCK_CONFIG, MOCK_SYSTEM_PROMPT)
    key2 = get_cache_key(MOCK_JD_PATH, MOCK_JD_TEXT, MOCK_CONFIG, MOCK_SYSTEM_PROMPT)
    assert key1 == key2
    assert key1 == EXPECTED_CACHE_KEY

def test_get_cache_key_sensitivity():
    """Tests that the cache key changes with different inputs."""
    key_base = get_cache_key(MOCK_JD_PATH, MOCK_JD_TEXT, MOCK_CONFIG, MOCK_SYSTEM_PROMPT)

    # Change JD text
    key_diff_text = get_cache_key(MOCK_JD_PATH, "Different JD text.", MOCK_CONFIG, MOCK_SYSTEM_PROMPT)
    assert key_base != key_diff_text

    # Change LLM model
    config_diff_llm = MOCK_CONFIG.copy()
    config_diff_llm['llm'] = {'model_name': 'other-llm', 'temperature': 0.5}
    key_diff_llm = get_cache_key(MOCK_JD_PATH, MOCK_JD_TEXT, config_diff_llm, MOCK_SYSTEM_PROMPT)
    assert key_base != key_diff_llm

    # Change Embedding model
    config_diff_emb = MOCK_CONFIG.copy()
    config_diff_emb['embedding'] = {'model_name': 'other-emb'}
    key_diff_emb = get_cache_key(MOCK_JD_PATH, MOCK_JD_TEXT, config_diff_emb, MOCK_SYSTEM_PROMPT)
    assert key_base != key_diff_emb

    # Change System Prompt
    key_diff_prompt = get_cache_key(MOCK_JD_PATH, MOCK_JD_TEXT, MOCK_CONFIG, "Different prompt.")
    assert key_base != key_diff_prompt


# --- Tests for get_cache_paths ---

def test_get_cache_paths():
    """Tests the generation of cache file paths."""
    cache_dir = Path("test_cache_dir")
    ideal_cv_path, embedding_path = get_cache_paths(cache_dir, EXPECTED_CACHE_KEY)
    expected_subdir = cache_dir / EXPECTED_CACHE_KEY[:2]
    assert ideal_cv_path == expected_subdir / f"{EXPECTED_CACHE_KEY}_ideal_cv.md"
    assert embedding_path == expected_subdir / f"{EXPECTED_CACHE_KEY}_embedding.npy"

# --- Tests for load_from_cache ---

@patch(f'{MODULE_PATH}.Path.is_file')
@patch(f'{MODULE_PATH}.load_text')
@patch(f'{MODULE_PATH}.np.load')
def test_load_from_cache_hit(mock_np_load, mock_load_text, mock_is_file):
    """Tests loading successfully from cache (cache hit)."""
    mock_is_file.return_value = True # Both files exist
    mock_load_text.return_value = MOCK_IDEAL_CV_TEXT
    mock_np_load.return_value = MOCK_EMBEDDING

    ideal_cv, embedding = load_from_cache(Path("dummy_cache"), EXPECTED_CACHE_KEY)

    assert ideal_cv == MOCK_IDEAL_CV_TEXT
    np.testing.assert_array_equal(embedding, MOCK_EMBEDDING)
    # Check that is_file was called (at least twice, for ideal_cv and embedding)
    assert mock_is_file.call_count >= 2
    # Verifying *which* specific Path object called is_file is complex with mocks.
    # This gets complex. Let's rely on the fact that load_text and np.load were called with the correct paths.
    mock_load_text.assert_called_once_with(EXPECTED_IDEAL_CV_PATH)
    mock_np_load.assert_called_once_with(EXPECTED_EMBEDDING_PATH)


@patch(f'{MODULE_PATH}.Path.is_file')
def test_load_from_cache_miss_file_not_found(mock_is_file):
    """Tests cache miss when files don't exist."""
    mock_is_file.return_value = False # Files don't exist

    ideal_cv, embedding = load_from_cache(Path("dummy_cache"), EXPECTED_CACHE_KEY)

    assert ideal_cv is None
    assert embedding is None
    mock_is_file.assert_called()


@patch(f'{MODULE_PATH}.Path.is_file')
@patch(f'{MODULE_PATH}.load_text', side_effect=Exception("Read error"))
def test_load_from_cache_read_error(mock_load_text, mock_is_file):
    """Tests cache miss due to error during file reading."""
    mock_is_file.return_value = True # Files exist

    ideal_cv, embedding = load_from_cache(Path("dummy_cache"), EXPECTED_CACHE_KEY)

    assert ideal_cv is None
    assert embedding is None
    mock_load_text.assert_called_once() # np.load shouldn't be called if text load fails


# --- Tests for save_to_cache ---

@patch(f'{MODULE_PATH}.Path.mkdir')
@patch(f'{MODULE_PATH}.np.save')
@patch('builtins.open', new_callable=mock_open)
def test_save_to_cache_success(mock_open_file, mock_np_save, mock_mkdir):
    """Tests saving successfully to cache."""
    cache_dir = Path("dummy_cache")

    save_to_cache(cache_dir, EXPECTED_CACHE_KEY, MOCK_IDEAL_CV_TEXT, MOCK_EMBEDDING)

    # Check directory creation - assert_called_once_with checks the arguments passed to mkdir
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # We implicitly test that the *correct* path called mkdir via the patch target and setup

    # Check text file write
    mock_open_file.assert_called_once_with(EXPECTED_IDEAL_CV_PATH, 'w', encoding='utf-8')
    mock_open_file().write.assert_called_once_with(MOCK_IDEAL_CV_TEXT)

    # Check numpy array save
    mock_np_save.assert_called_once()
    # Need to compare numpy arrays carefully
    call_args, _ = mock_np_save.call_args
    assert call_args[0] == EXPECTED_EMBEDDING_PATH
    np.testing.assert_array_equal(call_args[1], MOCK_EMBEDDING)


@patch(f'{MODULE_PATH}.Path.mkdir', side_effect=Exception("Disk full"))
def test_save_to_cache_mkdir_error(mock_mkdir):
    """Tests error during directory creation."""
    cache_dir = Path("dummy_cache")
    # We expect save_to_cache to catch the exception and print an error, not re-raise
    try:
        save_to_cache(cache_dir, EXPECTED_CACHE_KEY, MOCK_IDEAL_CV_TEXT, MOCK_EMBEDDING)
    except Exception as e:
        pytest.fail(f"save_to_cache raised an unexpected exception: {e}")
    mock_mkdir.assert_called_once()


@patch(f'{MODULE_PATH}.Path.mkdir')
@patch('builtins.open', new_callable=mock_open)
def test_save_to_cache_write_error(mock_open_file, mock_mkdir):
    """Tests error during file writing."""
    # Configure the mock file handle's write method to raise an error
    mock_file_handle = mock_open_file.return_value.__enter__.return_value
    mock_file_handle.write.side_effect = Exception("Write error")

    cache_dir = Path("dummy_cache")
    # We expect save_to_cache to catch the exception and print an error, not re-raise
    try:
        save_to_cache(cache_dir, EXPECTED_CACHE_KEY, MOCK_IDEAL_CV_TEXT, MOCK_EMBEDDING)
    except Exception as e:
        pytest.fail(f"save_to_cache raised an unexpected exception: {e}")

    mock_mkdir.assert_called_once()
    mock_open_file.assert_called_once_with(EXPECTED_IDEAL_CV_PATH, 'w', encoding='utf-8')
    mock_file_handle.write.assert_called_once() # Should attempt to write


# =========================================
# == Tests for Explanation Caching Funcs ==
# =========================================

# --- Tests for get_explanation_cache_key ---

def test_get_explanation_cache_key_consistency():
    """Tests explanation cache key consistency."""
    key1 = get_explanation_cache_key(
        EXPECTED_CACHE_KEY, MOCK_USER_CV_TEXT, MOCK_EXPLANATION_PROMPT_TEXT, MOCK_CONFIG
    )
    key2 = get_explanation_cache_key(
        EXPECTED_CACHE_KEY, MOCK_USER_CV_TEXT, MOCK_EXPLANATION_PROMPT_TEXT, MOCK_CONFIG
    )
    assert key1 == key2
    assert key1 == EXPECTED_EXPLANATION_CACHE_KEY

def test_get_explanation_cache_key_sensitivity():
    """Tests explanation cache key sensitivity to inputs."""
    key_base = get_explanation_cache_key(
        EXPECTED_CACHE_KEY, MOCK_USER_CV_TEXT, MOCK_EXPLANATION_PROMPT_TEXT, MOCK_CONFIG
    )

    # Change ideal CV key
    key_diff_ideal = get_explanation_cache_key(
        "different_ideal_key", MOCK_USER_CV_TEXT, MOCK_EXPLANATION_PROMPT_TEXT, MOCK_CONFIG
    )
    assert key_base != key_diff_ideal

    # Change user CV text
    key_diff_user_cv = get_explanation_cache_key(
        EXPECTED_CACHE_KEY, "Different user CV.", MOCK_EXPLANATION_PROMPT_TEXT, MOCK_CONFIG
    )
    assert key_base != key_diff_user_cv

    # Change explanation prompt
    key_diff_exp_prompt = get_explanation_cache_key(
        EXPECTED_CACHE_KEY, MOCK_USER_CV_TEXT, "Different explanation prompt.", MOCK_CONFIG
    )
    assert key_base != key_diff_exp_prompt

    # Change LLM model in config
    config_diff_llm = MOCK_CONFIG.copy()
    config_diff_llm['llm'] = {'model_name': 'other-llm-exp', 'temperature': 0.5}
    key_diff_llm = get_explanation_cache_key(
        EXPECTED_CACHE_KEY, MOCK_USER_CV_TEXT, MOCK_EXPLANATION_PROMPT_TEXT, config_diff_llm
    )
    assert key_base != key_diff_llm

# --- Tests for get_explanation_cache_path ---

def test_get_explanation_cache_path():
    """Tests the generation of the explanation cache file path."""
    cache_dir = Path("test_cache_dir_exp")
    exp_path = get_explanation_cache_path(cache_dir, EXPECTED_EXPLANATION_CACHE_KEY)
    expected_subdir = cache_dir / EXPECTED_EXPLANATION_CACHE_KEY[:2]
    assert exp_path == expected_subdir / f"{EXPECTED_EXPLANATION_CACHE_KEY}_explanation.txt"

# --- Tests for load_explanation_from_cache ---

@patch(f'{MODULE_PATH}.Path.is_file')
@patch(f'{MODULE_PATH}.load_text')
def test_load_explanation_from_cache_hit(mock_load_text, mock_is_file):
    """Tests loading explanation successfully from cache."""
    mock_is_file.return_value = True
    mock_load_text.return_value = MOCK_EXPLANATION_TEXT
    cache_dir = Path("dummy_cache") # Use a consistent dummy path

    explanation = load_explanation_from_cache(cache_dir, EXPECTED_EXPLANATION_CACHE_KEY)

    assert explanation == MOCK_EXPLANATION_TEXT
    mock_is_file.assert_called_once()
    # Construct the expected path based on the dummy cache dir used in the call
    expected_path = cache_dir / EXPECTED_EXPLANATION_CACHE_KEY[:2] / f"{EXPECTED_EXPLANATION_CACHE_KEY}_explanation.txt"
    mock_load_text.assert_called_once_with(expected_path)


@patch(f'{MODULE_PATH}.Path.is_file')
def test_load_explanation_from_cache_miss(mock_is_file):
    """Tests explanation cache miss when file doesn't exist."""
    mock_is_file.return_value = False
    cache_dir = Path("dummy_cache")

    explanation = load_explanation_from_cache(cache_dir, EXPECTED_EXPLANATION_CACHE_KEY)

    assert explanation is None
    mock_is_file.assert_called_once()


@patch(f'{MODULE_PATH}.Path.is_file')
@patch(f'{MODULE_PATH}.load_text', side_effect=Exception("Read error"))
def test_load_explanation_from_cache_read_error(mock_load_text, mock_is_file):
    """Tests explanation cache miss due to read error."""
    mock_is_file.return_value = True
    cache_dir = Path("dummy_cache")

    explanation = load_explanation_from_cache(cache_dir, EXPECTED_EXPLANATION_CACHE_KEY)

    assert explanation is None
    mock_is_file.assert_called_once()
    mock_load_text.assert_called_once() # Should attempt to load

# --- Tests for save_explanation_to_cache ---

@patch(f'{MODULE_PATH}.Path.mkdir')
@patch('builtins.open', new_callable=mock_open)
def test_save_explanation_to_cache_success(mock_open_file, mock_mkdir):
    """Tests saving explanation successfully."""
    cache_dir = Path("dummy_cache_exp")
    # Construct the expected path based on the dummy cache dir used in the call
    expected_path = cache_dir / EXPECTED_EXPLANATION_CACHE_KEY[:2] / f"{EXPECTED_EXPLANATION_CACHE_KEY}_explanation.txt"


    save_explanation_to_cache(cache_dir, EXPECTED_EXPLANATION_CACHE_KEY, MOCK_EXPLANATION_TEXT)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_open_file.assert_called_once_with(expected_path, 'w', encoding='utf-8')
    mock_open_file().write.assert_called_once_with(MOCK_EXPLANATION_TEXT)


@patch(f'{MODULE_PATH}.Path.mkdir', side_effect=Exception("Disk full"))
def test_save_explanation_to_cache_mkdir_error(mock_mkdir):
    """Tests error during explanation cache directory creation."""
    cache_dir = Path("dummy_cache_exp")
    try:
        save_explanation_to_cache(cache_dir, EXPECTED_EXPLANATION_CACHE_KEY, MOCK_EXPLANATION_TEXT)
    except Exception as e:
        pytest.fail(f"save_explanation_to_cache raised an unexpected exception: {e}")
    mock_mkdir.assert_called_once()


@patch(f'{MODULE_PATH}.Path.mkdir')
@patch('builtins.open', new_callable=mock_open)
def test_save_explanation_to_cache_write_error(mock_open_file, mock_mkdir):
    """Tests error during explanation file writing."""
    mock_file_handle = mock_open_file.return_value.__enter__.return_value
    mock_file_handle.write.side_effect = Exception("Write error")
    cache_dir = Path("dummy_cache_exp")
    # Construct the expected path based on the dummy cache dir used in the call
    expected_path = cache_dir / EXPECTED_EXPLANATION_CACHE_KEY[:2] / f"{EXPECTED_EXPLANATION_CACHE_KEY}_explanation.txt"

    try:
        save_explanation_to_cache(cache_dir, EXPECTED_EXPLANATION_CACHE_KEY, MOCK_EXPLANATION_TEXT)
    except Exception as e:
        pytest.fail(f"save_explanation_to_cache raised an unexpected exception: {e}")

    mock_mkdir.assert_called_once()
    mock_open_file.assert_called_once_with(expected_path, 'w', encoding='utf-8')
    mock_file_handle.write.assert_called_once() # Should attempt to write
