# DDIB Toolkit - Agent Instructions

## Developer Commands
- `make requirements` - Install dependencies (uses uv sync)
- `make install-hooks` - Install pre-commit hooks
- `make lint` - Run ruff, pylint, ty
- `make format` - Format with ruff (fix + format)
- `make test` - Run pytest
- Order: format -> lint -> test

## Package Setup
- Uses `uv` for package management (not pip directly)
- Uses `src` layout: package is `src/ddib/`
- Python 3.14+ required
- PyTorch uses CUDA 128 index on Windows/Linux

## Code Style
- Line length: 99
- Quote style: single quotes
- Indent: tabs (not spaces)
- Use `uv run python` for running scripts (see Makefile)

## Pre-commit
- Runs ruff (fix + format) and ty type checker
- Local `ty` hook requires `uv run ty check`