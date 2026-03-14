#!/usr/bin/env python3
"""
Aggregate RULER benchmark results from JSONL files and generate CSV summaries.

Called by scripts/run_ruler.sh after all GPU processes complete.

Usage:
    python scripts/summarize_ruler.py \
        --models /path/to/model1 /path/to/model2 \
        --method full --sparse_budget 2048 --rank 160 --chunk_size 8 \
        --datalens 4096 8192 131072
"""

import argparse
import csv
import glob
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize RULER benchmark results")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths")
    parser.add_argument("--method", required=True)
    parser.add_argument("--sparse_budget", required=True)
    parser.add_argument("--rank", required=True)
    parser.add_argument("--chunk_size", required=True)
    parser.add_argument("--datalens", nargs="+", type=int, required=True)
    return parser.parse_args()


def collect_results(archive_dir, datalen, method, sparse_budget, rank, chunk_size):
    """Parse JSONL result files for a given model + datalen configuration."""
    suffix = f"_{datalen}_{method}_{sparse_budget}_{rank}_{chunk_size}.jsonl"
    pattern = f"{archive_dir}/ruler/*{suffix}"
    files = sorted(glob.glob(pattern))

    results = []
    for f in files:
        task = os.path.basename(f).replace(suffix, "")
        scores = []
        null_count = 0
        total_count = 0
        with open(f) as fh:
            for line in fh:
                data = json.loads(line)
                scores.extend(data.get("correct", []))
                for pred in data.get("prediction", []):
                    total_count += 1
                    if not pred or pred.strip() == "":
                        null_count += 1
        if scores:
            avg = sum(scores) / len(scores)
            results.append((task, len(scores), avg, null_count, total_count))

    return results


def check_oom(archive_dir, datalen, method, sparse_budget, rank, chunk_size):
    """Check if an OOM marker exists for this configuration."""
    marker = f"{archive_dir}/ruler/.oom_{datalen}_{method}_{sparse_budget}_{rank}_{chunk_size}"
    return os.path.exists(marker)


def print_terminal_table(model_short, method, all_results, oom_datalens):
    """Print markdown summary table to terminal."""
    print(f"## {model_short} (Method: {method})")
    print()

    all_datalens = sorted(set(list(all_results.keys()) + list(oom_datalens)))
    for datalen in all_datalens:
        print(f"### DataLen: {datalen}")
        print()

        if datalen in oom_datalens and datalen not in all_results:
            print("**OOM** — CUDA out of memory")
            print()
            continue

        if datalen in all_results:
            results = all_results[datalen]
            print("| Dataset | Samples | Accuracy | Nulls |")
            print("|:--------|--------:|---------:|------:|")
            total_score = 0
            for task, samples, acc, nulls, total in results:
                print(f"| ruler/{task} | {samples} | {acc:.4f} | {nulls}/{total} |")
                total_score += acc
            avg_all = total_score / len(results)
            print(f"| **Average** | - | **{avg_all:.4f}** | - |")
            if datalen in oom_datalens:
                print("> (partial, OOM on some GPUs)")
            print()


def write_csv(archive_dir, datalen, method, sparse_budget, rank, chunk_size,
              results=None, is_oom=False):
    """Write CSV result file for a model × datalen combination."""
    csv_path = (
        f"{archive_dir}/ruler/"
        f"results_{datalen}_{method}_{sparse_budget}_{rank}_{chunk_size}.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        if is_oom and not results:
            writer.writerow(["", 0])
            writer.writerow(["Tasks", "ALL"])
            writer.writerow(["Score", "OOM"])
            writer.writerow(["Nulls", "OOM"])
            print(f"CSV saved: {csv_path} (OOM)")
        elif results:
            tasks = [r[0] for r in results]
            scores = [round(r[2] * 100, 2) for r in results]
            nulls = [f"{r[3]}/{r[4]}" for r in results]

            writer.writerow([""] + list(range(len(tasks))))
            writer.writerow(["Tasks"] + tasks)
            writer.writerow(["Score"] + scores)
            writer.writerow(["Nulls"] + nulls)
            print(f"CSV saved: {csv_path}")

    return csv_path


def main():
    args = parse_args()

    for model in args.models:
        model_short = os.path.basename(model)
        archive_dir = f"archive/{model_short}"

        all_results = {}
        oom_datalens = set()

        for datalen in args.datalens:
            if check_oom(archive_dir, datalen, args.method,
                         args.sparse_budget, args.rank, args.chunk_size):
                oom_datalens.add(datalen)

            results = collect_results(
                archive_dir, datalen, args.method,
                args.sparse_budget, args.rank, args.chunk_size
            )
            if results:
                all_results[datalen] = results

        if not all_results and not oom_datalens:
            continue

        # Terminal output
        print_terminal_table(model_short, args.method, all_results, oom_datalens)

        # CSV output
        all_datalens = sorted(set(list(all_results.keys()) + list(oom_datalens)))
        for datalen in all_datalens:
            is_oom = datalen in oom_datalens
            results = all_results.get(datalen)
            write_csv(
                archive_dir, datalen, args.method,
                args.sparse_budget, args.rank, args.chunk_size,
                results=results,
                is_oom=is_oom and not results,
            )

        print()


if __name__ == "__main__":
    main()
