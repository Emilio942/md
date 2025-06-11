# Makefile for ProteinMD development automation
.PHONY: help install install-dev test test-integration lint format type-check security clean build docs docker release

# Default target
help:
	@echo "ProteinMD Development Automation"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  install         Install ProteinMD in production mode"
	@echo "  install-dev     Install ProteinMD in development mode with all tools"
	@echo "  test           Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-all       Run all tests with coverage"
	@echo "  lint           Run all linting tools"
	@echo "  format         Format code with black and isort"
	@echo "  type-check     Run type checking with mypy"
	@echo "  security       Run security checks"
	@echo "  quality        Run all quality checks (lint, type, security)"
	@echo "  clean          Clean up build artifacts and cache"
	@echo "  build          Build distribution packages"
	@echo "  docs           Build documentation"
	@echo "  docker         Build Docker images"
	@echo "  docker-dev     Start development Docker environment"
	@echo "  release        Create a new release"
	@echo "  pre-commit     Install and run pre-commit hooks"
	@echo ""

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,analysis,docs]"
	pre-commit install

# Testing targets
test:
	pytest proteinMD/tests/ -v --tb=short

test-integration:
	python scripts/run_integration_tests.py

test-all:
	pytest proteinMD/tests/ --cov=proteinMD --cov-report=html --cov-report=term --cov-report=xml

test-quick:
	pytest proteinMD/tests/ -x --tb=short -q

# Code quality targets
format:
	black proteinMD/
	isort proteinMD/

lint:
	flake8 proteinMD/

type-check:
	mypy proteinMD/ --ignore-missing-imports

security:
	bandit -r proteinMD/ --skip B101,B601
	safety check

quality: format lint type-check security
	@echo "All quality checks completed"

# Documentation
docs:
	cd docs && sphinx-build -b html . _build/html
	@echo "Documentation built in docs/_build/html"

docs-live:
	cd docs && sphinx-autobuild -b html . _build/html --host 0.0.0.0 --port 8000

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

build-wheel:
	python -m build --wheel

build-sdist:
	python -m build --sdist

# Docker targets
docker:
	docker build -t proteinmd:latest .

docker-dev:
	docker-compose up -d proteinmd-dev

docker-jupyter:
	docker-compose up -d proteinmd-jupyter
	@echo "Jupyter Lab available at http://localhost:8888"

docker-docs:
	docker-compose up -d proteinmd-docs
	@echo "Documentation available at http://localhost:8000"

docker-stop:
	docker-compose down

# Pre-commit hooks
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Release management
release-patch:
	bump2version patch

release-minor:
	bump2version minor

release-major:
	bump2version major

release-check:
	python -m twine check dist/*

# Development utilities
dev-setup: install-dev pre-commit
	@echo "Development environment setup complete"

dev-test: format lint test
	@echo "Development tests completed"

# Performance profiling
profile:
	python -m cProfile -o profile.prof scripts/performance_profile.py
	python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"

# Memory profiling
memory-profile:
	python -m memory_profiler scripts/memory_profile.py

# Benchmark runs
benchmark:
	python utils/benchmark_comparison.py --all-systems --output-dir benchmark_results

# Database management (if applicable)
db-reset:
	rm -f proteinmd.db
	python -c "from proteinMD.database import init_db; init_db()"

# Analysis utilities
analyze-code:
	radon cc proteinMD/ -a
	radon mi proteinMD/
	xenon --max-average A --max-modules A --max-absolute B proteinMD/

# Git utilities
git-clean:
	git clean -fdx

git-reset:
	git reset --hard HEAD
	git clean -fdx

# Environment management
venv:
	python -m venv venv
	@echo "Activate with: source venv/bin/activate"

conda-env:
	conda env create -f environment.yml
	@echo "Activate with: conda activate proteinmd"

# Quick start for new developers
quickstart: venv install-dev pre-commit test
	@echo ""
	@echo "✅ Quick start completed successfully!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source venv/bin/activate"
	@echo "2. Run tests: make test"
	@echo "3. Start developing!"
	@echo ""

# Continuous Integration simulation
ci-test: quality test-all
	@echo "CI pipeline simulation completed"

# Release preparation
release-prep: clean quality test-all build
	@echo "Release preparation completed"
	@echo "Ready for release!"

# Full pipeline (what CI runs)
pipeline: install-dev quality test-all build
	@echo "Full pipeline completed successfully"
