import argparse
import json
import random
from pathlib import Path

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    baseline_rows = read_jsonl(args.baseline)
    candidate_rows = read_jsonl(args.candidate)

    baseline_by_id = {row["id"]: row for row in baseline_rows}
    candidate_by_id = {row["id"]: row for row in candidate_rows}

    common_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    if not common_ids:
        raise ValueError("No overlapping ids found between baseline and candidate files.")

    pairwise_rows = []
    for ex_id in common_ids:
        b = baseline_by_id[ex_id]
        c = candidate_by_id[ex_id]

        baseline_text = b.get("generated_text", b.get("prediction", b.get("output", "")))
        candidate_text = c.get("generated_text", c.get("prediction", c.get("output", "")))

        if not baseline_text or not candidate_text:
            continue

        swap = rng.random() < 0.5
        if swap:
            candidate_a = candidate_text
            candidate_b = baseline_text
            a_model = "finetuned"
            b_model = "baseline"
        else:
            candidate_a = baseline_text
            candidate_b = candidate_text
            a_model = "baseline"
            b_model = "finetuned"

        pairwise_rows.append({
            "id": ex_id,
            "messages": b.get("messages") or c.get("messages"),
            "reference_response": b.get("reference_response") or c.get("reference_response"),
            "metadata": b.get("metadata", {}),
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "a_model": a_model,
            "b_model": b_model
        })

    write_jsonl(args.output, pairwise_rows)
    print(f"Wrote {len(pairwise_rows)} pairwise rows to {args.output}")

if __name__ == "__main__":
    main()
