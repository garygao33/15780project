import argparse
import json
from pathlib import Path


FIELDS = [
    "persuasiveness",
    "clarity",
    "evidence_use",
    "respectfulness",
    "safety",
    "overall",
]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows):
    n = len(rows)
    sums = {k: 0.0 for k in FIELDS}

    for row in rows:
        judge = row["judge"]
        for k in FIELDS:
            sums[k] += judge[k]

    means = {f"{k}_mean": (sums[k] / n if n else 0.0) for k in FIELDS}
    return {"num_examples": n, **means}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input_file)
    summary = summarize(rows)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
