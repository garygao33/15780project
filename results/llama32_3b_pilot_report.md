# Llama 3.2 3B Pilot Results

## Training Setup

- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Method: QLoRA with LoRA rank `16`, alpha `32`, dropout `0.05`
- Train split: `data/processed/train.jsonl`
- Training subset: `6000` shuffled examples from `60638` training examples
- Max sequence length: `1024`
- Microbatch size: `4`
- Gradient accumulation: `4`
- Effective batch size: `16`
- Max steps: `300`
- Precision: `bf16`
- Optimizer: `paged_adamw_8bit`
- W&B run: `https://wandb.ai/garygao/persuasion-llm/runs/ia5in48u`

## Training Metrics

- Runtime: `1216.9181` seconds
- Train samples per second: `3.944`
- Train steps per second: `0.247`
- Final train loss: `1.6975366083780925`
- Final epoch fraction: `0.8`

Logged training loss:

| Step | Loss |
| ---: | ---: |
| 25 | 1.8738 |
| 50 | 1.7616 |
| 75 | 1.6881 |
| 100 | 1.6847 |
| 125 | 1.6852 |
| 150 | 1.6799 |
| 175 | 1.6755 |
| 200 | 1.6890 |
| 225 | 1.6708 |
| 250 | 1.6441 |
| 275 | 1.6324 |
| 300 | 1.6852 |

## Sample Evaluation

Evaluation was run on a 50-example held-out sample from `data/processed/test.jsonl`.

| Model | ROUGE-1 F1 | ROUGE-L F1 | Avg Prediction Chars |
| --- | ---: | ---: | ---: |
| Base Llama 3.2 3B | 0.3290 | 0.1557 | 818.92 |
| Fine-tuned pilot adapter | 0.3573 | 0.1735 | 817.10 |

The pilot adapter improved reference-overlap metrics on the sample. These metrics do not directly measure persuasiveness, so LLM-as-judge evaluation is still needed for the main project claim.

## Next Recommended Runs

- Run LLM-as-judge on baseline and pilot sample predictions.
- Train a longer run with validation enabled and checkpoint selection.
- Compare training on `data/processed` versus `data/processed_success`.
- Add direction-specific instructions into training prompts so the chatbot can explicitly increase or decrease belief depending on the task.
