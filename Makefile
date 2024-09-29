#################################################################################
# GLOBALS                                                                       #
#################################################################################
ENVNAME := .venv
VENV := $(ENVNAME)/bin

PROJECT_NAME = lstm_autoencoder
PYTHON_INTERPRETER = $(VENV)/python


#################################################################################
# COMMANDS                                                                      #
#################################################################################
.PHONY: run
run:
	$(PYTHON_INTERPRETER) src/lstm_autoencoder/main.py

.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -empty -delete

	rm -rf .*_cache
	rm -rf logs
	rm -rf site

	rm -rf data/02_intermediate/*
	rm -rf data/03_primary/*
	rm -rf data/04_feature/*
	rm -rf data/05_model_input/*
	rm -rf data/06_model/*
	rm -rf data/07_model_output/*
	rm -rf data/08_reporting/*

.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests
	$(VENV)/coverage report

.PHONY: lint
lint:
	git add --intent-to-add .
	. $(VENV)/activate; $(VENV)/pre-commit run --all-files

#################################################################################
# SETUP
#################################################################################

.PHONY: update
update:
	@echo "[template] Updating the data science project template..."
	copier update --trust --defaults --conflict inline
	@echo "[template] If necessary, solve all existing merge conflicts manually"
	@echo "[template] If necessary, update Poetry lockfile with 'poetry lock --no-update'"
	@echo "[template] Execute 'make install' to update all dependencies"

.PHONY: install
install: initialize_git install_global_dependencies install_python_and_dependencies install_precommit
	@echo "[template] Install ipython kernel..."
	$(VENV)/python -m ipykernel install --user --name $(PROJECT_NAME)
	@echo "[template] Initilize direnv..."
	direnv allow

.PHONY: initialize_git
initialize_git:
	@echo "[template] Initialize git..."
	git init --quiet

.PHONY: install_global_dependencies
install_global_dependencies:
	@echo "[template] Install global dependencies..."
	@sh ./scripts/install_global_dependencies.sh
	@sh ./scripts/check_pyenv_python_version_availability.sh


.PHONY: install_python_and_dependencies
install_python_and_dependencies:
	@echo "[template] Install python..."
	pyenv install -s

	@echo "[template] Configure poetry..."
	poetry env use $(shell pyenv root)/versions/3.11.9/bin/python

	@echo "[template] Install dependencies..."
	poetry install --all-extras --no-interaction


.PHONY: install_precommit
install_precommit:
	@echo "[template] Install pre-commit hooks..."
	$(VENV)/pre-commit install
	$(VENV)/pre-commit install-hooks
	$(VENV)/pre-commit install --hook-type commit-msg
