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

## Install pre-commit hooks
.PHONY: install-hooks
install-hooks: requirements
	uv run pre-commit install

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff, pylint and ty (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff check
	pylint src/ddib/
	uv run ty check src/ddib/

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
train: data
	$(PYTHON_INTERPRETER) -m src.experiments.modeling.train


## Run optimized grid search (advanced version - recommended)
.PHONY: resnet-vgg
resnet-vgg: data
	$(PYTHON_INTERPRETER) -m src.experiments.modeling.advanced_optimized_grid_search_train \
		--config config/grid_search_config.yaml \
		--results-dir results/grid_search \
		--max-concurrent -1 \
		--batch-size 2 \
		--logdir tb_logs_grid

.PHONY: efficientnet-experiments
efficientnet: data
	$(PYTHON_INTERPRETER) -m src.experiments.modeling.advanced_optimized_grid_search_train \
		--config config/efficientnet_config.yaml \
		--results-dir results/efficientnet_b2 \
		--max-concurrent -1 \
		--batch-size 4 \
		--logdir tb_logs_efficientnet

.PHONY: baseline-experiments
baseline: data
	$(PYTHON_INTERPRETER) -m src.experiments.modeling.advanced_optimized_grid_search_train \
		--config config\baseline_config.yaml \
		--results-dir results/baseline \
		--max-concurrent -1 \
		--batch-size 4 \
		--logdir tb_logs_baseline

## Download and prepare CIFAR-10-C dataset for robustness evaluation
.PHONY: cifar10c-data
cifar10c-data:
	$(PYTHON_INTERPRETER) -m src.experiments.robustness.prepare_data \
		-d data/processed \
		--cleanup

## Verify CIFAR-10-C data integrity
.PHONY: verify-cifar10c
verify-cifar10c:
	$(PYTHON_INTERPRETER) -m src.experiments.robustness.prepare_data verify \
		--data-dir data/processed/CIFAR-10-C

## Run robustness validation (train and evaluate all models from config)
.PHONY: robustness
robustness: cifar10c-data
	$(PYTHON_INTERPRETER) -m src.experiments.robustness.validate \
		--config config/robustness_config.yaml \
		--data-dir data/processed \
		--output-dir results/robustness

## Run robustness validation with parallel training (uses all GPUs)
.PHONY: robustness-parallel
robustness-parallel: cifar10c-data
	$(PYTHON_INTERPRETER) -m src.experiments.robustness.validate \
		--config config/robustness_config.yaml \
		--data-dir data/processed \
		--output-dir results/robustness \
		--max-concurrent -1

## Run robustness validation for a single existing model
.PHONY: robustness-single
robustness-single: cifar10c-data
	$(PYTHON_INTERPRETER) -m src.experiments.robustness.validate \
		--config config/robustness_config.yaml \
		--model-path models/best_model.pt \
		--model-arch vgg11 \
		--bottleneck-width 2048 \
		--data-dir data/processed \
		--output-dir results/robustness

## Run robustness evaluation only (skip training if model exists)
.PHONY: robustness-eval
robustness-eval: cifar10c-data
	$(PYTHON_INTERPRETER) -m src.experiments.robustness.validate \
		--config config/robustness_config.yaml \
		--data-dir data/processed \
		--output-dir results/robustness \
		--skip-training

## Run robustness evaluation only (skip training if model exists)
.PHONY: robustness-eval-parallel
robustness-eval-parallel: cifar10c-data
	$(PYTHON_INTERPRETER) -m src.experiments.robustness.validate \
		--config config/robustness_config.yaml \
		--data-dir data/processed \
		--output-dir results/robustness \
		--skip-training

## Run robustness with skip for both existing models and existing results
.PHONY: robustness-skip-existing
robustness-skip-existing: cifar10c-data
	$(PYTHON_INTERPRETER) -m src.experiments.robustness.validate \
		--config config/robustness_config.yaml \
		--data-dir data/processed \
		--output-dir results/robustness \
		--skip-training \
		--skip-validated

## Run robustness analysis notebook
.PHONY: robustness-analysis
robustness-analysis:
	$(PYTHON_INTERPRETER) notebooks/robustness_validation_analysis.py


#################################################################################
# ANALYSIS COMMANDS                                                              #
#################################################################################

## Run unified grid search analysis (ResNet + VGG): visualizations + numerical
.PHONY: analyze-results
analyze-results:
	$(PYTHON_INTERPRETER) notebooks/analyze_grid_search.py

## Run unified EfficientNet analysis: visualizations + numerical
.PHONY: analyze-efficientnet
analyze-efficientnet:
	$(PYTHON_INTERPRETER) notebooks/analyze_efficientnet.py

## Run all analysis scripts
.PHONY: analyze-all
analyze-all: analyze-results analyze-efficientnet

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
