import pytest

from rank_jobs import load_config  # Import the module directly

# --- Fixtures ---


@pytest.fixture
def valid_config_path(tmp_path):
    """Creates a valid temporary config file."""
    content = """
cv_path: "data/cv/test_cv.md"
jd_dir: "data/jd_test"
llm:
  model_name: "test-llm-model"
  temperature: 0.1
embedding:
  model_name: "test-embedding-model"
prompts:
  system_message_file: "prompts/test_system.txt"
  explanation_prompt_file: "prompts/test_explanation.txt" # Added
cache:
  enabled: true
  directory: "test_cache"
  explanation_threshold: 0.5 # Added
"""
    p = tmp_path / "config_valid.yaml"
    p.write_text(content)
    return p


@pytest.fixture
def invalid_yaml_path(tmp_path):
    """Creates an invalid temporary YAML file."""
    content = "key: value\nkey_no_colon"
    p = tmp_path / "config_invalid_yaml.yaml"
    p.write_text(content)
    return p


@pytest.fixture
def missing_keys_config_path(tmp_path):
    """Creates a temporary config file with missing keys."""
    content = """
cv_path: "data/cv/test_cv.md"
# jd_dir: "data/jd_test" # Missing jd_dir
llm:
  model_name: "test-llm-model"
  # temperature: 0.1 # Missing temperature
embedding:
  model_name: "test-embedding-model"
prompts:
  system_message_file: "prompts/test_system.txt"
cache:
  enabled: true
  # directory: "test_cache" # Missing directory
"""
    p = tmp_path / "config_missing_keys.yaml"
    p.write_text(content)
    return p


# --- Tests ---


def test_load_config_valid(valid_config_path):
    """Tests loading a valid configuration file."""
    config = load_config(valid_config_path)
    assert config["cv_path"] == "data/cv/test_cv.md"
    assert config["llm"]["model_name"] == "test-llm-model"
    assert config["cache"]["enabled"] is True
    assert config["cache"]["directory"] == "test_cache"


def test_load_config_not_found():
    """Tests loading a non-existent configuration file."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")


def test_load_config_invalid_yaml(invalid_yaml_path):
    """Tests loading an invalid YAML file."""
    with pytest.raises(ValueError, match="Error parsing YAML"):
        load_config(invalid_yaml_path)


def test_load_config_missing_keys(missing_keys_config_path):
    """Tests loading a config file with missing required keys."""
    with pytest.raises(ValueError, match="Config file is missing required"):
        load_config(missing_keys_config_path)
