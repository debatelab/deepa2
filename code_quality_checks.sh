#!/bin/bash
echo "===================================================================="
echo "Running pylint..."
poetry run pylint deepa2datasets tests

echo "Running flake8..."
poetry run flake8 deepa2datasets tests

echo "Running mypy..."
poetry run mypy deepa2datasets tests

echo "Running black..."
poetry run black --diff --color deepa2datasets tests

