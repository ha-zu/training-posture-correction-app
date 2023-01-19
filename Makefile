.PHONY: black clean flake8 help isort lint-all run setup

# =========================================
# Base settings
# =========================================
PROJECT_NAME := Training posture correction App
PYTHON := python

SRC_DIR := training-posture-correction-app
APP := $(SRC_DIR)/posture_app.py

# =========================================
# Help command lists
# =========================================
help: ## Show help
	@echo "usage: make <target>"
	@echo ""
	@echo "target:"
	@grep -E '^[a-zA-Z0123456789-]+:.?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.?## "}; {printf "\033[36m%-25s\033[0m%s\n", $$1, $$2}'
	@echo ""

# =========================================
# Linters
# =========================================
black: ## Run black
	@git ls-files '*.py' | xargs $(PYTHON) -m black

flake8: ## Run flake8
	@git ls-files '*.py' | xargs $(PYTHON) -m flake8 --exit-zero

isort: ## Run isort
	@git ls-files '*.py' | xargs $(PYTHON) -m isort

lint-all: ## Run all linters
	@make flake8
	@make  isort

# =========================================
# Run main file
# =========================================
run: ## Run posture_app.py
	$(PYTHON) $(APP)

setup: ## Install requirements.txt
	pip install -r requirements.txt

clean: ## Clean __pycache__
	@find  $(SRC_DIR)/ -name '__pycache__' -exec rm -rf {} \;
