################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
import gc
def colored(text, color):
    colors = {'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m', 'blue': '\033[94m', 'cyan': '\033[96m'}
    return f"{colors.get(color, '')}{text}\033[0m"
from argparse import ArgumentParser, Namespace

from data.dataset import Dataset
os.chdir(root_dir)

from models import choose_model_class

dataset_name = "ruler/qa_2"

configs = {
    "gradientai/Llama-3-8B-Instruct-Gradient-1048k": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 8,
            "shadowkv_bsz": 48,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 4,
            "shadowkv_bsz": 24,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 12,
        }
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "4k": {
            "sparse_budget": 128,
            "min_prompt_len": 4096,
            "baseline_bsz": 1,
            "shadowkv_bsz": 1,
        },
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 8,
            "shadowkv_bsz": 48,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 4,
            "shadowkv_bsz": 24,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 12,
        }
    },
    "/home/zijie/models/Llama-3.1-8B-Instruct": {
        "4k": {
            "sparse_budget": 128,
            "min_prompt_len": 4096,
            "baseline_bsz": 1,
            "shadowkv_bsz": 1,
        },
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 8,
            "shadowkv_bsz": 48,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 4,
            "shadowkv_bsz": 24,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 12,
        }
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 8,
            "shadowkv_bsz": 48,
        }
    },

    "01-ai/Yi-9B-200K": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 10,
            "shadowkv_bsz": 42,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 5,
            "shadowkv_bsz": 21,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 10,
        }
    },
    "THUDM/glm-4-9b-chat-1m": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 12,
            "shadowkv_bsz": 50,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 6,
            "shadowkv_bsz": 25,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 3,
            "shadowkv_bsz": 12,
        }
    }
}


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", choices=["gradientai/Llama-3-8B-Instruct-Gradient-1048k", "meta-llama/Meta-Llama-3.1-8B-Instruct", "01-ai/Yi-9B-200K","THUDM/glm-4-9b-chat-1m", "meta-llama/Llama-2-7b-chat-hf", "/home/zijie/models/Llama-3.1-8B-Instruct"])
    p.add_argument("--model_path", type=str, default=None, help="Local path overriding the model_name huggingface download path.")
    p.add_argument("--datalen", type=str, default="122k", choices=["4k", "60k", "122k", "244k"])

    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()

    model_name = args.model_name
    model_path = args.model_path if args.model_path is not None else model_name
    length = args.datalen

    min_prompt_len = configs[model_name][length]["min_prompt_len"]
    temperature = 0.6
    baseline_bsz = configs[model_name][length]["baseline_bsz"]
    shadowkv_bsz = configs[model_name][length]["shadowkv_bsz"]
    sparse_budget = configs[model_name][length]["sparse_budget"]


    ##################### Baseline #####################
    LLM = choose_model_class(model_name)
    llm = LLM(model_name=model_path, device='cuda:0',  batch_size=baseline_bsz, max_length=min_prompt_len, attn_mode='full', sparse_budget=sparse_budget)
    
    # Map requested datalen to actual generated RULER lengths
    dataset_name = "ruler/niah_single_1"
    datalen_map = {"4k": 4096, "60k": 65536, "122k": 131072, "244k": 262144}
    ruler_len = datalen_map.get(length, 262144)
    dataset = Dataset(dataset_name, llm.tokenizer, ruler_len, 20)

    # Gather prompts
    input_ids_list = [dataset[i][0] for i in range(llm.batch_size)]
    actual_min_len = min(t.shape[-1] for t in input_ids_list)
    min_prompt_len = min(min_prompt_len, actual_min_len)
    input_ids = torch.cat([t[:, :min_prompt_len] for t in input_ids_list], dim=0)

    assert input_ids.shape[-1] == min_prompt_len

    _, throughput_baseline = llm.batch_generate(input_ids.to(llm.device), gen_len=100, benchmark=True, temperature=temperature)
    print(colored(f"[Baseline] Throughput: {throughput_baseline} tokens/s", 'red'))

    del llm.kv_cache
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    ##################### ShadowKV #####################
    LLM = choose_model_class(model_name)
    llm = LLM(model_name=model_path, device='cuda:0',  batch_size=shadowkv_bsz, max_length=min_prompt_len, attn_mode='shadowkv_cpu', sparse_budget=sparse_budget)
    dataset = Dataset(dataset_name, llm.tokenizer, ruler_len, 100)

    input_ids_list = [dataset[i][0] for i in range(llm.batch_size)]
    actual_min_len = min(t.shape[-1] for t in input_ids_list)
    min_prompt_len = min(min_prompt_len, actual_min_len)
    input_ids = torch.cat([t[:, :min_prompt_len] for t in input_ids_list], dim=0)

    assert input_ids.shape[-1] == min_prompt_len

    _, throughput_shadowkv = llm.batch_generate(input_ids.to(llm.device), gen_len=100, benchmark=True, temperature=temperature)
    print(colored(f"[ShadowKV] Throughput: {throughput_shadowkv} tokens/s", 'red'))
    
    print(colored(f"\n[{model_name} Datalen {length}]\nBaseline Throughput: {throughput_baseline} tokens/s\nShadowKV Throughput: {throughput_shadowkv} tokens/s\n", 'green'))