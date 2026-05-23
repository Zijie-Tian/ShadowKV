################################################################################
#
# Worker script for Table 4 reproduction.
# Runs ONE (method, batch_size, context_length) throughput test and outputs
# a JSON result to stdout.
#
# Usage:
#   python test/e2e_single_run.py \
#     --model_name "/home/zijie/models/Llama-3.1-8B-Instruct" \
#     --method full --batch_size 2 --datalen 16k
#
################################################################################

import os
import sys
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
os.chdir(root_dir)

import torch
import gc
from argparse import ArgumentParser

from data.dataset import Dataset
from models import choose_model_class


# Supported context lengths → (prompt tokens, sparse_budget, ruler_dataset_length)
# ruler_dataset_length must be one of: 4096, 8192, 16384, 32768, 65536, 131072, 262144
DATALEN_CONFIGS = {
    "4k":   {"prompt_len": 1024 * 4,   "sparse_budget": 128,  "ruler_len": 4096},
    "8k":   {"prompt_len": 1024 * 8,   "sparse_budget": 128,  "ruler_len": 8192},
    "16k":  {"prompt_len": 1024 * 16,  "sparse_budget": 256,  "ruler_len": 16384},
    "32k":  {"prompt_len": 1024 * 32,  "sparse_budget": 512,  "ruler_len": 32768},
    "48k":  {"prompt_len": 1024 * 48,  "sparse_budget": 768,  "ruler_len": 65536},
    "60k":  {"prompt_len": 1024 * 60,  "sparse_budget": 1024, "ruler_len": 65536},
    "64k":  {"prompt_len": 1024 * 64,  "sparse_budget": 1024, "ruler_len": 65536},
    "80k":  {"prompt_len": 1024 * 80,  "sparse_budget": 1280, "ruler_len": 131072},
    "96k":  {"prompt_len": 1024 * 96,  "sparse_budget": 1536, "ruler_len": 131072},
    "122k": {"prompt_len": 1024 * 122, "sparse_budget": 2048, "ruler_len": 131072},
    "244k": {"prompt_len": 1024 * 244, "sparse_budget": 4096, "ruler_len": 262144},
    "488k": {"prompt_len": 1024 * 488, "sparse_budget": 8192, "ruler_len": 524288},
}


def parse_args():
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, required=True,
                   help="Model name (used for choose_model_class and config lookup)")
    p.add_argument("--model_path", type=str, default=None,
                   help="Local path to model weights (overrides model_name for loading)")
    p.add_argument("--method", type=str, required=True, choices=["full", "shadowkv"])
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--datalen", type=str, required=True, choices=list(DATALEN_CONFIGS.keys()))
    p.add_argument("--gen_len", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    model_name = args.model_name
    model_path = args.model_path if args.model_path else model_name
    batch_size = args.batch_size
    datalen = args.datalen
    method = args.method
    gen_len = args.gen_len
    device = args.device

    cfg = DATALEN_CONFIGS[datalen]
    min_prompt_len = cfg["prompt_len"]
    sparse_budget = cfg["sparse_budget"]
    ruler_len = cfg["ruler_len"]
    temperature = 0.6

    attn_mode = method
    if method == "shadowkv" and batch_size != 1:
        raise ValueError("shadowkv uses the batch_size=1 ShadowKVCache path; the CPU offload cache path has been removed")

    # Build model
    LLM = choose_model_class(model_name)
    llm = LLM(
        model_name=model_path,
        device=device,
        batch_size=batch_size,
        max_length=min_prompt_len,
        attn_mode=attn_mode,
        sparse_budget=sparse_budget,
    )

    # Load dataset — we use RULER niah_single_1 data and truncate to desired length
    dataset_name = "ruler/niah_single_1"
    num_samples = max(batch_size, 100)
    dataset = Dataset(dataset_name, llm.tokenizer, ruler_len, num_samples)

    # Gather prompts and truncate to min_prompt_len
    input_ids_list = [dataset[i][0] for i in range(batch_size)]
    actual_min_len = min(t.shape[-1] for t in input_ids_list)
    prompt_len = min(min_prompt_len, actual_min_len)
    input_ids = torch.cat([t[:, :prompt_len] for t in input_ids_list], dim=0)

    assert input_ids.shape[0] == batch_size
    assert input_ids.shape[1] == prompt_len

    # Run throughput benchmark
    _, throughput = llm.batch_generate(
        input_ids.to(llm.device),
        gen_len=gen_len,
        benchmark=True,
        temperature=temperature,
    )

    # Output result as JSON to stdout (the orchestrator parses this)
    result = {"throughput": round(throughput, 2)}
    print(f"@@RESULT@@{json.dumps(result)}@@END@@")

    # Cleanup
    del llm.kv_cache
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
