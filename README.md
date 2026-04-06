# Run Guide

## 1. Create the environment

```bash
CONDA_PKGS_DIRS=$PWD/.conda-pkgs conda create -y -p $PWD/.conda python=3.10
conda activate $PWD/.conda
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```bash
OPENAI_API_KEY=...
OPENAI_JUDGE_MODEL=gpt-5.2
HF_TOKEN=...
```

## 2. Put the dataset files in place

Expected files:

- `data/train.jsonl`
- `data/validation.jsonl`
- `data/test.jsonl`

Format reference:

- [data/README.md](/home/gary/Desktop/15780/project/data/README.md)

## 3. Validate the dataset

```bash
python scripts/run.py validate --input-file data/train.jsonl --require-assistant
python scripts/run.py validate --input-file data/validation.jsonl --require-assistant
python scripts/run.py validate --input-file data/test.jsonl
```

## 4. Update config paths if needed

Edit:

- [configs/train.yaml](/home/gary/Desktop/15780/project/configs/train.yaml)
- [configs/generate.yaml](/home/gary/Desktop/15780/project/configs/generate.yaml)
- [configs/judge.yaml](/home/gary/Desktop/15780/project/configs/judge.yaml)

## 5. Train

```bash
python scripts/run.py train --config configs/train.yaml
```

## 6. Generate predictions

```bash
python scripts/run.py generate --config configs/generate.yaml
```

## 7. Run automatic evaluation

```bash
python scripts/run.py evaluate \
  --input-file outputs/predictions/test_predictions.jsonl \
  --output-file outputs/predictions/metrics.json
```

## 8. Run LLM-as-a-judge evaluation

```bash
python scripts/run.py judge --config configs/judge.yaml
```

## Outputs

- Adapter checkpoints: `outputs/llama31-8b-persuasion/`
- Predictions: `outputs/predictions/test_predictions.jsonl`
- Automatic metrics: `outputs/predictions/metrics.json`
- Judge scores: `outputs/judge/judged_predictions.jsonl`
- Judge summary: `outputs/judge/summary.json`
