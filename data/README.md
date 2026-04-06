# Dataset contract

No dataset is included in this repository. Convert the selected data source into JSONL files matching one of the schemas below.

## Recommended schema: chat-style records

Each line is one JSON object:

```json
{
  "id": "example-001",
  "messages": [
    {"role": "system", "content": "You are a respectful persuasive assistant."},
    {"role": "user", "content": "I do not think recycling matters because one person cannot change anything."},
    {"role": "assistant", "content": "One person alone does not solve the problem, but shared habits scale through communities and policy support."}
  ],
  "metadata": {
    "topic": "recycling",
    "split": "train"
  }
}
```

For training, the final message must be an `assistant` response.

For evaluation or generation, the final assistant message is optional:

```json
{
  "id": "example-101",
  "messages": [
    {"role": "system", "content": "You are a respectful persuasive assistant."},
    {"role": "user", "content": "I am not convinced public transit investment is worth it."}
  ],
  "reference_response": "Transit can reduce congestion and widen access to jobs, which is why cities often treat it as shared infrastructure."
}
```

## Alternate flat schema

The loader also accepts a flat schema:

```json
{
  "id": "example-flat-1",
  "system_prompt": "You are a respectful persuasive assistant.",
  "user_prompt": "I do not think voting matters.",
  "assistant_response": "Close elections and local races are often decided by surprisingly small margins."
}
```

## Files expected by the scaffold

- `train.jsonl`: supervised finetuning data
- `validation.jsonl`: optional validation data used during training
- `test.jsonl`: prompts for generation and evaluation

## Important note

Review dataset licensing and ethics constraints before training. If the raw data is not already in one of these schemas, add a conversion step before running the pipeline.
