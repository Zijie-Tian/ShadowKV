import argparse
import time
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Standalone test for Tensor Parallelism (TP) measuring prefill performance")
    parser.add_argument("--model_name", type=str, default="/home/zijie/models/Llama-3.1-8B-Instruct", help="Path or HuggingFace ID of the model")
    parser.add_argument("--input_len", type=int, default=4000, help="Length of generated input prompt to simulate context")
    parser.add_argument("--tp", type=int, default=2, help="Tensor Parallelism size (number of GPUs)")
    args = parser.parse_args()

    print(f"Loading model {args.model_name} with vLLM (Tensor Parallelism TP={args.tp})...")

    # Initialize vLLM inference engine
    # Setting tensor_parallel_size natively distributes the Q, K, V, Out and MLP matrices
    # across the GPUs and automatically manages the NCCL AllReduce communications.
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=16000,
        enforce_eager=False,  # Let vLLM use cuda graphs for maximum performance
        trust_remote_code=True,
        gpu_memory_utilization=0.9
    )

    # We only care about prefill, so limit generation to 1 token output.
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

    print(f"\nPre-warming generating fake inputs of length {args.input_len}...")
    # Generate a repeating word to form a long context (vLLM v1 uses text prompts)
    prompt = "hello " * (args.input_len // 6 + 1)

    # Warmup
    print("Executing warmup pass...")
    _ = llm.generate(prompts=["test"], sampling_params=sampling_params, use_tqdm=False)

    print(f"Starting Prefill Performance Test for ~{args.input_len} tokens in TP={args.tp}...")

    start_time = time.time()

    # This automatically runs the prefill compute utilizing distributed TP algorithms
    outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params, use_tqdm=False)

    end_time = time.time()

    # Because we only generated 1 output token, the time is >99% entirely dominated by the prefill calculation
    prefill_time = end_time - start_time
    total_tokens = args.input_len

    print(f"\nPrefill Completed in {prefill_time:.4f} seconds.")
    print(f"Prefill Throughput: {total_tokens / prefill_time:.2f} tokens/sec")

    metrics = outputs[0].metrics
    if metrics is not None and getattr(metrics, 'first_token_time', None) and getattr(metrics, 'time_in_queue', None):
        ttft = metrics.first_token_time - metrics.arrival_time - metrics.time_in_queue
        print(f"vLLM engine internal TTFT (Time To First Token/Pure Prefill): {ttft:.4f} seconds")
        print(f"vLLM internal Throughput: {total_tokens / ttft:.2f} tokens/sec")

if __name__ == "__main__":
    main()
