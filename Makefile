PYTHON ?= python

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

validate-train:
	$(PYTHON) scripts/run.py validate --input-file data/examples/train.jsonl --require-assistant

train:
	$(PYTHON) scripts/run.py train --config configs/train.yaml

generate:
	$(PYTHON) scripts/run.py generate --config configs/generate.yaml

evaluate:
	$(PYTHON) scripts/run.py evaluate --input-file outputs/predictions/test_predictions.jsonl --output-file outputs/predictions/metrics.json

judge:
	$(PYTHON) scripts/run.py judge --config configs/judge.yaml
