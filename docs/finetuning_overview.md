# Finetuning Overview

This project finetunes open-weight chat models to produce persuasive responses. The goal is not to teach the model English from scratch. The base model already knows how to write and follow instructions. Finetuning nudges it toward a narrower behavior: given a topic, a user's current belief, and a desired persuasion direction, generate a clear argument meant to move the user's belief in that direction.

## What The Data Teaches

Each dataset row is converted into a chat-style training example. The prompt tells the model the persuasion task, and the target answer is the argument written for that case.

The important field is `direction`:

- `positive` means the argument is intended to increase belief in the claim.
- `negative` means the argument is intended to decrease belief in the claim.

This helps the model learn direction-aware persuasion. A generic chat model may give a balanced explanation or list pros and cons. Our finetuned model sees many examples where the response is supposed to argue toward a specific belief shift, so it should become more consistent at producing targeted persuasive replies.

One limitation is that the data is mostly one-turn persuasion. That means the model is learning how to write the next persuasive response, not a full long-term dialogue strategy. For a chatbot, we can still use it turn by turn: take the user's latest stance, choose the target direction, and generate the next persuasive response.

## Instruction And Target

For supervised finetuning, every example has two parts:

- The instruction is the prompt given to the model. It includes the topic, available context, belief information, and the intended direction.
- The target is the assistant response from the dataset: the persuasive argument the model should learn to produce.

During training, the model sees the full chat-formatted text, but the loss is applied only to the assistant response tokens. In plain terms, we do not reward the model for copying the prompt. We reward it for predicting the persuasive answer after reading the prompt.

## Finetuning Method

We use QLoRA, which is a memory-efficient version of finetuning:

- The base model is loaded in 4-bit precision to reduce GPU memory use.
- The original base model weights stay frozen.
- Small trainable LoRA adapter layers are added to important projection layers inside the model.
- Training updates only those adapter weights.

The final saved model is therefore an adapter, not a full copy of the base model. To use it later, we load the original base model and then load the adapter on top.

For the completed Llama run, teammates need two pieces:

- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Adapter directory: `outputs/llama32-3b-persuasion-full`

The generation config `configs/generate_llama32_3b_full_sample.yaml` shows how the code loads both pieces together. The `base_model_name` field points to the original model, and `adapter_path` points to the finetuned adapter.

## Current Models

The completed main run uses `meta-llama/Llama-3.2-3B-Instruct`. A second comparison run uses `Qwen/Qwen2.5-3B-Instruct`, which is a different model family in a similar size range.

Using a second family is useful for the presentation because it helps separate two questions:

- Does persuasion finetuning help compared with the base model?
- Does the result depend on the model family?

Both runs use the same processed dataset and a similar QLoRA setup so the comparison is reasonably fair.

## Hyperparameters

The main settings are intentionally conservative:

- One epoch: enough to adapt to the task without spending too much compute or heavily overfitting.
- `1024` token context: long enough for the prompt and argument while keeping training stable.
- LoRA rank `16`: a common balance between adapter capacity and memory use.
- Learning rate `2e-4`: typical for QLoRA adapter training.
- Effective batch size `16`: stable for the 3B runs on the available GPUs.

These settings are not claimed to be optimal. They are a reasonable first full-run setup for comparing models in a class project.

## Evaluation

Evaluation has two parts:

- Automatic metrics compare generated responses with reference responses using ROUGE. This is useful as a sanity check, but it does not fully measure persuasion because a good argument can use different wording from the reference.
- LLM-as-a-judge asks a stronger model to rate outputs for persuasiveness, clarity, evidence use, respectfulness, and safety. This is more aligned with the project goal because persuasiveness is subjective.

The main comparison should be run on the same held-out prompts for the base model and the finetuned model.
