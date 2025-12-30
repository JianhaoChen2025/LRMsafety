import os
import sys
import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.moderation import load_llamaguard, moderate

# Add the evaluation directory to the path to find concurrent_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent_utils import run_tasks_concurrently

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="jailbreakbench_eval.csv")
    parser.add_argument("--concurrency", type=int, default=3, help="Maximum number of concurrent workers.")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file.")
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", 'judge_comparison', split="test", token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto").eval()
    model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    def generate(prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
    dataset_list = list(dataset)

    # Resume functionality: load existing results if output file exists
    existing_df = None
    processed_prompts = set()
    if os.path.exists(args.output):
        print(f"Found existing output file: {args.output}")
        existing_df = pd.read_csv(args.output)
        processed_prompts = set(existing_df["prompt"].astype(str))
        print(f"Found {len(processed_prompts)} already processed prompts")
    
    # Filter out already processed examples
    remaining_examples = [ex for ex in dataset_list if ex["prompt"] not in processed_prompts]
    print(f"Processing {len(remaining_examples)} remaining examples (out of {len(dataset_list)} total)")

    def process_prompt(_, example):
        prompt = example["prompt"]
        response = generate(prompt)
        return {"prompt": prompt, "response": response}

    new_rows = [
        r for r in run_tasks_concurrently(
            remaining_examples,
            process_prompt,
            max_workers=args.concurrency,
            desc="Generating JBB Responses",
        ) if r is not None
    ]
    
    # Combine existing and new results
    if existing_df is not None and len(new_rows) > 0:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    elif existing_df is not None:
        df = existing_df
    elif len(new_rows) > 0:
        df = pd.DataFrame(new_rows)
    else:
        df = pd.DataFrame(columns=["prompt", "response"])
    
    # Moderation with resume support
    tokenizer_m, model_m = load_llamaguard()

    def moderate_row(_, item):
        idx, sample = item
        flag = moderate(sample["prompt"], sample["response"], tokenizer_m, model_m)
        return idx, flag

    # Filter out already moderated rows
    rows_to_moderate = []
    for idx, row in df.iterrows():
        if pd.isna(row.get("response_flag")) or row.get("response_flag") == "":
            rows_to_moderate.append((idx, row))
    
    if rows_to_moderate:
        print(f"Moderating {len(rows_to_moderate)} remaining rows (out of {len(df)} total)")
        moderation_results = run_tasks_concurrently(
            rows_to_moderate,
            moderate_row,
            max_workers=args.concurrency,
            desc="Moderating outputs",
        )
        for result in moderation_results:
            if result is None:
                continue
            row_idx, flag = result
            df.at[row_idx, "response_flag"] = flag
    else:
        print("All rows already moderated")
    
    safe_count = 0
    for _, row in df.iterrows():
        if row.get("response_flag") == "safe":
            safe_count += 1
    df.to_csv(args.output, index=False)
    print(f"Safe@1: {(safe_count / len(df)) * 100:.2f}%")
    print(f"Saved flagged output to {args.output}")

if __name__ == "__main__":
    main()
