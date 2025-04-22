# List available recipes
default:
    @just --list

# Install project dependencies
install:
    uv sync --all-groups
    bash -c "source .venv/bin/activate"

# Run job-rank application
rank *ARGS='':
    python src/rank_jobs.py {{ARGS}}

# Run pytest
test *ARGS='tests':
    pytest {{ARGS}}

# Run type checking
typecheck *ARGS='src tests':
    mypy {{ARGS}}

# Run linting checks
lint *ARGS='src tests':
    ruff check {{ARGS}}

# Format code, sort imports, and fix issues
format *ARGS='src tests':
    ruff format {{ARGS}}
    ruff check --select I --fix {{ARGS}}
    ruff check --fix {{ARGS}}

# Run all checks (tests, type checking, linting)
check: (test) (typecheck) (lint)

# Format code and run all checks
format-check: (format) (check)

# Remove all cached data
clean-cache:
    rm -rf cache/*

# Remove only explanation caches
clean-explanations:
    find cache -name '*_explanation.txt' -delete

# Display cache statistics
cache-stats:
    find cache -type f -name '*.md' | wc -l | xargs echo "Cached Ideal CVs:"
    find cache -type f -name '*.npy' | wc -l | xargs echo "Cached Embeddings:"
    find cache -type f -name '*_explanation.txt' | wc -l | xargs echo "Cached Explanations:"
