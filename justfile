default:
    @just --list

# Install dependencies
install:
    uv sync --all-groups
