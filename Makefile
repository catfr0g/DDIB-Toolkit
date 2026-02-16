#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ddib
PYTHON_VERSION = 3.15
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff, pylint and mypy (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff check
	pylint src/ddib/
	mypy src/ddib/

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"


.PHONY: tensorboard
tensorboard: requirements
	uv run tensorboard --logdir tb_logs


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m src.experiments.dataset


.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) -m src.experiments.modeling.train


## Run optimized grid search (advanced version - recommended)
.PHONY: grid-search
grid-search: requirements
	$(PYTHON_INTERPRETER) -m src.experiments.modeling.advanced_optimized_grid_search_train \
		--config config/grid_search_config.yaml \
		--results-dir results/grid_search \
		--max-concurrent -1 \
		--batch-size 2


## Run optimized grid search (batch version)
.PHONY: grid-search-batch
grid-search-batch: requirements
	$(PYTHON_INTERPRETER) -m src.experiments.modeling.batch_optimized_grid_search_train \
		--config config/grid_search_config.yaml \
		--results-dir results/grid_search \
		--batch-size 4


## Run optimized grid search (parallel version)
.PHONY: grid-search-parallel
grid-search-parallel: requirements
	$(PYTHON_INTERPRETER) -m src.experiments.modeling.optimized_grid_search_train \
		--config config/grid_search_config.yaml \
		--results-dir results/grid_search \
		--max-concurrent -1


## Run GPU-optimized grid search (single experiment at a time for maximum GPU utilization)
.PHONY: grid-search-gpu
grid-search-gpu: requirements
	$(PYTHON_INTERPRETER) -m src.experiments.modeling.gpu_optimized_grid_search_train \
		--config config/grid_search_config.yaml \
		--results-dir results/grid_search \
		--max-concurrent 3

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
