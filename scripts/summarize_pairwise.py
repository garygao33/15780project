import argparse
import json
import math
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows):
    baseline_wins = 0
    finetuned_wins = 0
    ties = 0

    for row in rows:
        winner = row.get("winning_model")
        if winner == "baseline":
            baseline_wins += 1
        elif winner == "finetuned":
            finetuned_wins += 1
        else:
            ties += 1

    total = len(rows)
    non_ties = baseline_wins + finetuned_wins

    baseline_win_rate_all = baseline_wins / total if total else 0.0
    finetuned_win_rate_all = finetuned_wins / total if total else 0.0
    tie_rate = ties / total if total else 0.0

    baseline_win_rate_no_ties = baseline_wins / non_ties if non_ties else 0.0
    finetuned_win_rate_no_ties = finetuned_wins / non_ties if non_ties else 0.0

    bt_score = math.log((finetuned_wins + 0.5) / (baseline_wins + 0.5))

    return {
        "num_examples": total,
        "baseline_wins": baseline_wins,
        "finetuned_wins": finetuned_wins,
        "ties": ties,
        "baseline_win_rate_all": baseline_win_rate_all,
        "finetuned_win_rate_all": finetuned_win_rate_all,
        "tie_rate": tie_rate,
        "baseline_win_rate_excluding_ties": baseline_win_rate_no_ties,
        "finetuned_win_rate_excluding_ties": finetuned_win_rate_no_ties,
        "bradley_terry_logodds_finetuned_vs_baseline": bt_score,
    }


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
