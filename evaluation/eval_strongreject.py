import os
import sys
import argparse
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.moderation import load_llamaguard, moderate

# Add the evaluation directory to the path to find concurrent_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent_utils import run_tasks_concurrently

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="strongreject_eval.csv")
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--concurrency", type=int, default=3, help="Maximum number of concurrent workers.")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file.")

    # Use the public dataset or a local CSV as needed
    dataset = load_dataset("walledai/StrongREJECT", token=hf_token)["train"]
    prompts = dataset["prompt"][:args.n]

    # Resume functionality: load existing results if output file exists
    existing_df = None
    processed_prompts = set()
    if os.path.exists(args.output):
        print(f"Found existing output file: {args.output}")
        existing_df = pd.read_csv(args.output)
        processed_prompts = set(existing_df["prompt"].astype(str))
        print(f"Found {len(processed_prompts)} already processed prompts")
    
    # Filter out already processed prompts
    remaining_prompts = [p for p in prompts if str(p) not in processed_prompts]
    print(f"Processing {len(remaining_prompts)} remaining prompts (out of {len(prompts)} total)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    model.to(device).eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gen_kwargs = dict(
        do_sample=False,
        temperature=0.0,
        top_p=1.00,
        max_new_tokens=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.3
    )

    def process_prompt(_, prompt):
        if not prompt:
            return None
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outs = model.generate(**inputs, **gen_kwargs)
        texts = tokenizer.batch_decode(outs, skip_special_tokens=True)
        response = texts[0][len(prompt):].strip()
        torch.cuda.empty_cache()
        return {"prompt": prompt, "response": response}

    new_rows = [
        r for r in run_tasks_concurrently(
            remaining_prompts,
            process_prompt,
            max_workers=args.concurrency,
            desc="Generating responses",
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
        idx, row = item
        flag = moderate(row["prompt"], row["response"], tokenizer_m, model_m)
        return idx, flag

    # Filter out already moderated rows
    rows_to_moderate = []
    for idx, row in df.iterrows():
        if pd.isna(row.get("response_flag")) or row.get("response_flag") == "":
            rows_to_moderate.append((idx, row))
    
    safe1_count = 0
    if rows_to_moderate:
        print(f"Moderating {len(rows_to_moderate)} remaining rows (out of {len(df)} total)")
        moderation_results = run_tasks_concurrently(
            rows_to_moderate,
            moderate_row,
            max_workers=args.concurrency,
            desc="Evaluating",
        )
        for result in moderation_results:
            if result is None:
                continue
            row_idx, flag = result
            df.at[row_idx, "response_flag"] = flag
    else:
        print("All rows already moderated")
    
    for _, row in df.iterrows():
        if row.get("response_flag") == "safe":
            safe1_count += 1
    df.to_csv(args.output, index=False)
    print(f"Safe@1: {(safe1_count / len(df)) * 100:.2f}%")
    print(f"Saved flagged output to {args.output}")

if __name__ == "__main__":
    main()
