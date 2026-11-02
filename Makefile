.PHONY: install run run-tracked mlflow-ui test lint

install:
	pip install -e ".[dev,tracking]"

run:              ## simulate, train, evaluate -> reports/ (~2 min)
	python -m fraud.pipeline

run-tracked:      ## same, logging every model to MLflow (mlflow.db)
	python -m fraud.pipeline --mlflow

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

test:
	pytest -q

lint:
	ruff check src tests
	ruff format --check src tests
