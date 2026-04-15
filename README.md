# Run Guide

## Setup

```bash
CONDA_PKGS_DIRS=$PWD/.conda-pkgs conda create -y -p $PWD/.conda python=3.10
conda activate $PWD/.conda
python -m pip install -r requirements.txt
cp .env.example .env
```

Add tokens to `.env`:

```bash
HF_TOKEN=...
OPENAI_API_KEY=...
OPENAI_JUDGE_MODEL=gpt-5.2
```

## Prepare Data

Put the teammate CSV at:

```bash
data/master_persuasion_studies_analytic_filtered.csv
```

Convert it:

```bash
python scripts/run.py prepare-data \
  --input-csv data/master_persuasion_studies_analytic_filtered.csv \
  --output-dir data/processed \
  --keep directional
```

Validate:

```bash
python scripts/run.py validate --input-file data/processed/train.jsonl --require-assistant
python scripts/run.py validate --input-file data/processed/validation.jsonl --require-assistant
python scripts/run.py validate --input-file data/processed/test.jsonl
```

## Train Models

```bash
PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run.py train --config configs/train_llama32_3b_full.yaml
```

Adapter output:

```bash
outputs/llama32-3b-persuasion-full
```

Second model-family comparison:

```bash
PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run.py train --config configs/train_qwen25_3b_full.yaml
```

Adapter output:

```bash
outputs/qwen25-3b-persuasion-full
```

## Generate

Full test set:

```bash
python scripts/run.py generate --config configs/generate_llama32_3b_full.yaml
```

Small sample:

```bash
shuf -n 50 data/processed/test.jsonl > data/processed/test_sample_50.jsonl
python scripts/run.py generate --config configs/generate_llama32_3b_baseline_sample.yaml
python scripts/run.py generate --config configs/generate_llama32_3b_full_sample.yaml
```

## Use Shared Adapter

The finetuned output is a LoRA adapter, not a standalone model. To use a shared adapter zip, unzip it so the adapter files are under:

```bash
outputs/llama32-3b-persuasion-full
```

Then generate with:

```bash
python scripts/run.py generate --config configs/generate_llama32_3b_full_sample.yaml
```

This loads the base model `meta-llama/Llama-3.2-3B-Instruct` and then applies the adapter from `outputs/llama32-3b-persuasion-full`. A Hugging Face token with access to the base Llama model is still required in `.env`.

## Automatic Evaluation

```bash
python scripts/run.py evaluate \
  --input-file outputs/predictions/llama32-3b-baseline-sample.jsonl \
  --output-file outputs/predictions/llama32-3b-baseline-sample-metrics.json

python scripts/run.py evaluate \
  --input-file outputs/predictions/llama32-3b-full-sample.jsonl \
  --output-file outputs/predictions/llama32-3b-full-sample-metrics.json
```

## LLM Judge

```bash
python scripts/run.py judge --config configs/judge_llama32_3b_baseline_sample.yaml
python scripts/run.py judge --config configs/judge_llama32_3b_pilot_sample.yaml
```

## Project Notes

```bash
docs/finetuning_overview.md
results/current_progress.md
```
