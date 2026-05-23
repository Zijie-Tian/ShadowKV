#!/usr/bin/env python
"""Score ShadowKV LongBench prediction files.

The expected prediction layout mirrors LUTAttn's LongBench contract, but uses
ShadowKV's archive directory by default:

    archive/<model>/long_bench/<dataset>.jsonl
    archive/<model>/long_bench/result.json

Each JSONL row should contain: pred, answers, all_classes, and length.
"""

import argparse
import json
import os
from pathlib import Path
from statistics import mean

from data.longbench.metrics import (
    LONG_BENCH_LINE_TRUNCATE_TASKS,
    dataset2metric,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Score LongBench predictions")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name/path. Defaults to basename under archive/<model>/long_bench when --pred_dir is not set.")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Prediction directory containing <dataset>.jsonl files.")
    parser.add_argument("--archive_dir", type=str, default="archive",
                        help="Archive root used with --model. Default: archive")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated LongBench datasets to score, or all.")
    parser.add_argument("--e", action="store_true", help="Score LongBench-E length buckets")
    return parser.parse_args()


def model_to_dirname(model: str) -> str:
    return os.path.basename(model.rstrip("/"))


def resolve_pred_dir(args) -> Path:
    if args.pred_dir:
        return Path(args.pred_dir)
    if not args.model:
        raise ValueError("Either --pred_dir or --model is required")
    subdir = "long_bench_e" if args.e else "long_bench"
    return Path(args.archive_dir) / model_to_dirname(args.model) / subdir


def resolve_datasets(spec: str):
    if spec == "all":
        return list(dataset2metric.keys())
    return [item for item in spec.split(",") if item]


def score_prediction(dataset: str, prediction: str, answers, all_classes) -> float:
    if dataset in LONG_BENCH_LINE_TRUNCATE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    score = 0.0
    for answer in answers:
        score = max(score, dataset2metric[dataset](prediction, answer, all_classes=all_classes))
    return float(score)


def score_dataset(dataset: str, rows, longbench_e: bool):
    if longbench_e:
        buckets = {"0-4k": [], "4-8k": [], "8k+": []}
        for row in rows:
            length = int(row.get("length", 0))
            score = score_prediction(dataset, row["pred"], row["answers"], row.get("all_classes", []))
            if length < 4000:
                buckets["0-4k"].append(score)
            elif length < 8000:
                buckets["4-8k"].append(score)
            else:
                buckets["8k+"].append(score)
        return {name: round(100 * mean(vals), 2) if vals else 0.0 for name, vals in buckets.items()}

    scores = [score_prediction(dataset, row["pred"], row["answers"], row.get("all_classes", [])) for row in rows]
    return round(100 * mean(scores), 2) if scores else 0.0


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Compatibility with the previous ShadowKV batched output format.
            if "pred" not in data and "prediction" in data:
                preds = data.get("prediction", [])
                answers = data.get("ground_truth", [])
                classes = data.get("all_classes", [])
                lengths = data.get("length", [])
                for idx, pred in enumerate(preds):
                    rows.append({
                        "pred": pred,
                        "answers": answers[idx] if idx < len(answers) else [],
                        "all_classes": classes[idx] if idx < len(classes) else [],
                        "length": lengths[idx] if idx < len(lengths) else 0,
                    })
            else:
                rows.append(data)
    return rows


def main():
    args = parse_args()
    pred_dir = resolve_pred_dir(args)
    datasets = resolve_datasets(args.datasets)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    print(f"[Score] Prediction directory: {pred_dir}")
    print(f"[Score] Scoring script: score_longbench.py")
    print(f"[Score] Datasets: {','.join(datasets)}")

    scores = {}
    missing = []
    for dataset in datasets:
        path = pred_dir / f"{dataset}.jsonl"
        if not path.exists():
            missing.append(dataset)
            continue
        rows = load_jsonl(path)
        scores[dataset] = score_dataset(dataset, rows, args.e)
        print(f"[Score] {dataset}: {scores[dataset]} ({len(rows)} samples)")

    if missing:
        print(f"[Score] Missing datasets skipped: {','.join(missing)}")

    numeric_scores = [v for v in scores.values() if isinstance(v, (int, float))]
    if numeric_scores:
        scores["mean"] = round(mean(numeric_scores), 2)

    out_path = pred_dir / "result.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(f"[Score] Result file: {out_path}")


if __name__ == "__main__":
    main()
