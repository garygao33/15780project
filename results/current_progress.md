# Current Progress

## Completed Full Runs

Both completed models were trained with QLoRA adapters on the processed persuasion dataset.

| Model | Adapter | W&B Run | Train Loss | Runtime | Epoch |
| --- | --- | --- | ---: | ---: | ---: |
| Llama 3.2 3B Instruct | `outputs/llama32-3b-persuasion-full` | `https://wandb.ai/garygao/persuasion-llm/runs/3w0vkea4` | `0.3099` | `4029.6s` | `1.0` |
| Qwen 2.5 3B Instruct | `outputs/qwen25-3b-persuasion-full` | `https://wandb.ai/garygao/persuasion-llm/runs/0jc2pxno` | `1.5202` | `20043.1s` | `1.0` |

The Llama run was resumed from `checkpoint-3000` after an interruption. The final adapter completed the full epoch and saved successfully.

## Basic Sample Evaluation

Each model was evaluated on the same 50-example held-out test sample. This is a reference-overlap evaluation: it measures how similar the generated response is to the dataset's target response, not whether the response would actually persuade a person.

| Model | ROUGE-1 F1 | ROUGE-L F1 | Avg Prediction Chars |
| --- | ---: | ---: | ---: |
| Base Llama 3.2 3B | `0.3290` | `0.1557` | `818.92` |
| Fine-tuned Llama 3.2 3B | `0.3598` | `0.1779` | `806.22` |
| Base Qwen 2.5 3B | `0.3048` | `0.1494` | `791.34` |
| Fine-tuned Qwen 2.5 3B | `0.3458` | `0.1702` | `827.88` |

Both fine-tuned adapters improved over their base models on ROUGE. The Llama adapter improved ROUGE-1 by about `+0.0308` and ROUGE-L by about `+0.0222`. The Qwen adapter improved ROUGE-1 by about `+0.0410` and ROUGE-L by about `+0.0208`.

## Next Evaluation Step

ROUGE is useful as a quick sanity check, but the main presentation should also include the LLM-as-a-judge evaluation because it directly compares persuasiveness, clarity, and relevance of model responses.
