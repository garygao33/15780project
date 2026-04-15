# Current Progress

## Completed Full Run: Llama 3.2 3B

- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Output adapter: `outputs/llama32-3b-persuasion-full`
- Training data: `data/processed/train.jsonl`
- Validation data: `data/processed/validation.jsonl`
- Method: QLoRA with LoRA rank `16`, alpha `32`, dropout `0.05`
- Max sequence length: `1024`
- Effective batch size: `16`
- Epochs: `1`
- W&B run: `https://wandb.ai/garygao/persuasion-llm/runs/3w0vkea4`

Final saved metrics:

| Metric | Value |
| --- | ---: |
| Train loss | `0.3099` |
| Runtime | `4029.6` seconds |
| Train samples/sec | `15.048` |
| Train steps/sec | `0.941` |
| Epoch | `1.0` |

Note: this run was resumed from `checkpoint-3000` after an interruption. The adapter completed the full epoch and saved successfully.

## Completed Pilot Evaluation

A smaller pilot run was evaluated on a 50-example held-out sample.

| Model | ROUGE-1 F1 | ROUGE-L F1 | Avg Prediction Chars |
| --- | ---: | ---: | ---: |
| Base Llama 3.2 3B | `0.3290` | `0.1557` | `818.92` |
| Fine-tuned pilot adapter | `0.3573` | `0.1735` | `817.10` |

The pilot improved reference-overlap metrics, but ROUGE is only a weak proxy for persuasion. LLM-as-a-judge evaluation is the more relevant evaluation for the final presentation.

## Running Comparison Run: Qwen 2.5 3B

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Output adapter: `outputs/qwen25-3b-persuasion-full`
- Config: `configs/train_qwen25_3b_full.yaml`
- W&B run: `https://wandb.ai/garygao/persuasion-llm/runs/0jc2pxno`

This run uses the same processed dataset and similar QLoRA hyperparameters as the Llama 3B full run so that it can serve as a second model-family comparison.
