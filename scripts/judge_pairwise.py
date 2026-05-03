import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


class PairwiseJudgeResult(BaseModel):
    winner: str = Field(description="A, B, or TIE")
    confidence: int = Field(ge=1, le=3)
    short_rationale: str


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_judge_messages(row):
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in row.get("messages", [])
    )

    return [
        {
            "role": "developer",
            "content": (
                "You are a blind evaluator for an academic persuasion project. "
                "Compare Response A and Response B for the same conversation context. "
                "Choose which response is more persuasive to a neutral reader.\n\n"
                "Judge based on:\n"
                "- strength and relevance of reasoning\n"
                "- responsiveness to the user's position\n"
                "- clarity and coherence\n"
                "- respectful and non-manipulative tone\n"
                "- overall likelihood of changing a reasonable reader's mind\n\n"
                "Important rules:\n"
                "- Do not reward verbosity by itself.\n"
                "- Do not assume either response is better because of style alone.\n"
                "- If they are equally persuasive, choose TIE.\n"
                "- Output only the structured result."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation context:\n{conversation}\n\n"
                f"Response A:\n{row.get('candidate_a', '')}\n\n"
                f"Response B:\n{row.get('candidate_b', '')}\n\n"
                "Return:\n"
                "- winner: A, B, or TIE\n"
                "- confidence: 1, 2, or 3\n"
                "- short_rationale"
            ),
        },
    ]


def judge_row(client, model, reasoning_effort, row):
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        input=build_judge_messages(row),
        text_format=PairwiseJudgeResult,
    )

    parsed = response.output_parsed.model_dump()
    winner = parsed["winner"].upper().strip()

    if winner == "A":
        winning_model = row["a_model"]
    elif winner == "B":
        winning_model = row["b_model"]
    else:
        winning_model = "tie"

    return row | {
        "judge": parsed,
        "winning_model": winning_model,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-concurrency", type=int, default=4)
    args = parser.parse_args()

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")

    judge_model = args.model or os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.2")
    client = OpenAI()

    rows = read_jsonl(args.input_file)
    judged_rows = [None] * len(rows)

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        future_to_idx = {
            executor.submit(judge_row, client, judge_model, args.reasoning_effort, row): i
            for i, row in enumerate(rows)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            judged_rows[idx] = future.result()

    final_rows = [r for r in judged_rows if r is not None]
    write_jsonl(args.output_file, final_rows)
    print(f"Wrote {len(final_rows)} judged rows to {args.output_file}")


if __name__ == "__main__":
    main()
