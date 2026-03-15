#!/usr/bin/env python3
################################################################################
#
# Table 4 Reproduction: Generation Throughput Sweep
#
# Sweeps batch sizes across context lengths for both Full KV and ShadowKV.
# Supports multi-GPU parallel execution.
#
# Each (method, batch_size, datalen) test runs in a SEPARATE subprocess so that
# OOM errors are isolated and GPU memory is fully released between runs.
#
# Usage:
#   python test/e2e_sweep.py \
#     --model_name "gradientai/Llama-3-8B-Instruct-Gradient-1048k" \
#     --devices cuda:0 cuda:1
#
################################################################################

import os
import sys
import subprocess
import json
import re
import csv
import time
from argparse import ArgumentParser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

BATCH_SIZES = [2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48]
ALL_DATALENS = ["4k", "8k", "16k", "32k", "48k", "60k", "64k", "80k", "96k", "122k", "244k", "488k"]
DATALENS = ["16k", "32k", "48k", "64k", "80k", "96k"]
METHODS = ["full", "shadowkv_cpu"]

WORKER_SCRIPT = os.path.join(root_dir, "test", "e2e_single_run.py")


def parse_args():
    p = ArgumentParser(description="Table 4 Reproduction: Throughput Sweep")
    p.add_argument("--model_name", type=str, required=True,
                   help="Model name for choose_model_class")
    p.add_argument("--model_path", type=str, default=None,
                   help="Local path to model weights")
    p.add_argument("--devices", type=str, nargs="+", default=["cuda:0"],
                   help="GPU devices to use (e.g., cuda:0 cuda:1)")
    p.add_argument("--gen_len", type=int, default=100)
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-run timeout in seconds (default: 1800 = 30min)")
    p.add_argument("--methods", type=str, nargs="+", default=METHODS,
                   choices=METHODS, help="Methods to test")
    p.add_argument("--datalens", type=str, nargs="+", default=DATALENS,
                   choices=ALL_DATALENS, help="Context lengths to test")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES,
                   help="Batch sizes to test")
    p.add_argument("--output_dir", type=str, default=os.path.join(root_dir, "results"),
                   help="Directory to save result CSV")
    return p.parse_args()


def run_single_test(model_name, model_path, method, batch_size, datalen,
                    gen_len, device, timeout):
    """Spawn a subprocess for one test point. Returns throughput or 'OOM'."""

    cmd = [
        sys.executable, WORKER_SCRIPT,
        "--model_name", model_name,
        "--method", method,
        "--batch_size", str(batch_size),
        "--datalen", datalen,
        "--gen_len", str(gen_len),
        "--device", device,
    ]
    if model_path:
        cmd.extend(["--model_path", model_path])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "48"
    # Restrict CUDA visibility to the specific device index
    gpu_idx = device.split(":")[-1] if ":" in device else "0"
    env["CUDA_VISIBLE_DEVICES"] = gpu_idx

    # Override device to cuda:0 since CUDA_VISIBLE_DEVICES remaps
    cmd_override = []
    for i, c in enumerate(cmd):
        if c == "--device":
            cmd_override.append(c)
            # next arg will be the device, replace it
        elif i > 0 and cmd[i-1] == "--device":
            cmd_override.append("cuda:0")
        else:
            cmd_override.append(c)
    cmd = cmd_override

    label = f"[{device} | {method:12s} | bsz={batch_size:3d} | {datalen:4s}]"

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env=env, cwd=root_dir,
        )

        elapsed = time.time() - t0

        # Parse the result marker from stdout
        stdout = result.stdout
        match = re.search(r"@@RESULT@@(.+?)@@END@@", stdout)

        if result.returncode != 0:
            stderr_lower = (result.stderr or "").lower()
            stdout_lower = (result.stdout or "").lower()
            if "out of memory" in stderr_lower or "oom" in stderr_lower or \
               "out of memory" in stdout_lower:
                print(f"  {label}  OOM  ({elapsed:.0f}s)", flush=True)
                return "OOM"
            else:
                print(f"  {label}  FAILED (exit={result.returncode}, {elapsed:.0f}s)",
                      flush=True)
                if result.stderr:
                    for line in result.stderr.strip().split("\n")[-5:]:
                        print(f"    stderr: {line}")
                return "FAIL"

        if match:
            data = json.loads(match.group(1))
            tp = data["throughput"]
            print(f"  {label}  {tp:.2f} tok/s  ({elapsed:.0f}s)", flush=True)
            return tp
        else:
            print(f"  {label}  NO RESULT MARKER  ({elapsed:.0f}s)", flush=True)
            if stdout:
                for line in stdout.strip().split("\n")[-5:]:
                    print(f"    stdout: {line}")
            return "FAIL"

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  {label}  TIMEOUT ({elapsed:.0f}s)", flush=True)
        return "TIMEOUT"
    except Exception as e:
        print(f"  {label}  ERROR: {e}", flush=True)
        return "FAIL"


def format_table(results, batch_sizes, datalens, methods):
    """Print a formatted ASCII table matching Table 4 style."""

    col_width = 8
    header = f"{'Context':<10}" + "".join(f"{bs:>{col_width}}" for bs in batch_sizes)
    sep = "-" * len(header)

    for method in methods:
        method_label = "Full KV" if method == "full" else "ShadowKV"
        print()
        print(sep)
        print(f"{'':>10}{method_label:^{col_width * len(batch_sizes)}}")
        print(sep)
        print(header)
        print(sep)

        for dl in datalens:
            row = f"{dl.upper():<10}"
            for bs in batch_sizes:
                val = results.get((method, dl, bs), "N/A")
                if isinstance(val, (int, float)):
                    cell = f"{val:.2f}"
                else:
                    cell = str(val)
                row += f"{cell:>{col_width}}"
            print(row)
        print(sep)


def save_csv(results, batch_sizes, datalens, methods, output_path):
    """Save results as CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "context_length", "batch_size", "throughput_tok_s"])
        for method in methods:
            for dl in datalens:
                for bs in batch_sizes:
                    val = results.get((method, dl, bs), "N/A")
                    writer.writerow([method, dl, bs, val])

    print(f"\nResults saved to: {output_path}")


def run_method_sweep(method, datalens, batch_sizes, model_name, model_path,
                     gen_len, device, timeout, results, results_lock):
    """Run all (datalen, batch_size) tests for one method on one device."""
    method_label = "Full KV" if method == "full" else "ShadowKV"
    print(f"\n  [{device}] Starting method: {method_label}", flush=True)

    for dl in datalens:
        oom_hit = False
        print(f"\n  [{device}] Context Length: {dl.upper()} ({method_label})", flush=True)

        for bs in batch_sizes:
            if oom_hit:
                with results_lock:
                    results[(method, dl, bs)] = "OOM"
                print(f"  [{device} | {method:12s} | bsz={bs:3d} | {dl:4s}]  OOM (skipped)",
                      flush=True)
                continue

            tp = run_single_test(
                model_name=model_name,
                model_path=model_path,
                method=method,
                batch_size=bs,
                datalen=dl,
                gen_len=gen_len,
                device=device,
                timeout=timeout,
            )

            with results_lock:
                results[(method, dl, bs)] = tp

            if tp == "OOM":
                oom_hit = True


def main():
    args = parse_args()

    print("=" * 70)
    print("  Table 4 Reproduction: Generation Throughput Sweep")
    print("=" * 70)
    print(f"  Model:       {args.model_name}")
    print(f"  Model Path:  {args.model_path or '(same as model_name)'}")
    print(f"  Devices:     {args.devices}")
    print(f"  Methods:     {args.methods}")
    print(f"  Datalens:    {args.datalens}")
    print(f"  Batch Sizes: {args.batch_sizes}")
    print(f"  Gen Len:     {args.gen_len}")
    print(f"  Timeout:     {args.timeout}s per run")
    print("=" * 70)

    results = {}
    results_lock = threading.Lock()

    # Strategy: assign each method to a different GPU for parallel execution.
    # If there are more GPUs than methods, extra GPUs are unused.
    # If there are more methods than GPUs, methods share GPUs sequentially.
    method_device_pairs = []
    for i, method in enumerate(args.methods):
        device = args.devices[i % len(args.devices)]
        method_device_pairs.append((method, device))

    print(f"\n  GPU Assignment:")
    for method, device in method_device_pairs:
        label = "Full KV" if method == "full" else "ShadowKV"
        print(f"    {label:12s} -> {device}")
    print()

    # Run methods in parallel (one thread per method, each on its assigned GPU)
    with ThreadPoolExecutor(max_workers=len(method_device_pairs)) as executor:
        futures = []
        for method, device in method_device_pairs:
            future = executor.submit(
                run_method_sweep,
                method=method,
                datalens=args.datalens,
                batch_sizes=args.batch_sizes,
                model_name=args.model_name,
                model_path=args.model_path,
                gen_len=args.gen_len,
                device=device,
                timeout=args.timeout,
                results=results,
                results_lock=results_lock,
            )
            futures.append(future)

        # Wait for all to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  ERROR in thread: {e}", flush=True)

    # Print formatted table
    print("\n\n")
    print("=" * 70)
    print("  RESULTS: Table 4 — Generation Throughput (tokens/s)")
    print("=" * 70)
    format_table(results, args.batch_sizes, args.datalens, args.methods)

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"table4_{timestamp}.csv")
    save_csv(results, args.batch_sizes, args.datalens, args.methods, csv_path)


if __name__ == "__main__":
    main()
