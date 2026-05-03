
import argparse
import json
import math
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

def bt_score(rows):
    baseline_wins = 0
    finetuned_wins = 0

    for row in rows:
        winner = row.get("winning_model")
        if winner == "baseline":
            baseline_wins += 1
        elif winner == "finetuned":
            finetuned_wins += 1

    return math.log((finetuned_wins + 0.5) / (baseline_wins + 0.5))

def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.input_file)
    rng = random.Random(args.seed)

    point = bt_score(rows)

    scores = []
    for _ in range(args.n_bootstrap):
        sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        scores.append(bt_score(sample))

    summary = {
        "num_examples": len(rows),
        "bradley_terry_logodds": point,
        "bootstrap_95ci_lower": percentile(scores, 0.025),
        "bootstrap_95ci_upper": percentile(scores, 0.975),
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
