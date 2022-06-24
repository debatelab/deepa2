#!/bin/bash
echo "===================================================================="
echo "Running pylint..."
poetry run pylint --rcfile pyproject.toml deepa2 tests

echo "Running flake8..."
poetry run flake8 deepa2 tests

echo "Running mypy..."
poetry run mypy deepa2 tests

echo "Running black..."
poetry run black --diff --color deepa2 tests

