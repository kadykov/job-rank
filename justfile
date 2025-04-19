default:
    @just --list

# Install dependencies
install:
    uv sync --all-groups
    bash -c "source .venv/bin/activate"
