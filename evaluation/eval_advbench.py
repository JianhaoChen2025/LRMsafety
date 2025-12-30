import argparse
import os
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.moderation import load_llamaguard, moderate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Advbench.csv"),
        help="Path to Advbench CSV file containing a `question` column.",
    )
    parser.add_argument("--output", type=str, default="advbench_eval.csv", help="Path to write evaluation CSV.")
    parser.add_argument("--question-column", type=str, default="question", help="Column name for prompts/questions.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate for each question.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of questions to process concurrently when generating responses.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found at {args.input}")

    df_input = pd.read_csv(args.input)
    if args.question_column not in df_input.columns:
        raise ValueError(f"Column `{args.question_column}` not found in {args.input}.")

    existing_df = None
    completed_questions = set()
    if os.path.exists(args.output):
        try:
            existing_df = pd.read_csv(args.output)
            if args.question_column in existing_df.columns and "response" in existing_df.columns:
                completed_questions = set(existing_df[args.question_column].astype(str).tolist())
                if completed_questions:
                    print(f"[Resume] Loaded {len(completed_questions)} completed entries from {args.output}.")
            else:
                print(
                    f"[Resume] Existing output {args.output} missing `{args.question_column}` or `response` columns. "
                    "Ignoring for resume."
                )
                existing_df = None
        except Exception as exc:
            print(f"[Resume] Failed to read {args.output}: {exc}. Ignoring existing results.")
            existing_df = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    ).eval()
    model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def generate_batch(prompts: List[str]) -> List[str]:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_lengths = inputs["attention_mask"].sum(dim=1)
        responses = []
        for idx, output_ids in enumerate(outputs):
            gen_tokens = output_ids[int(input_lengths[idx]):]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            responses.append(text)
        return responses

    all_questions = df_input[args.question_column].tolist()
    normalized_question_indices = {str(question): idx for idx, question in enumerate(all_questions)}

    pending_questions = [q for q in all_questions if str(q) not in completed_questions]
    if completed_questions:
        skipped = len(all_questions) - len(pending_questions)
        print(f"[Resume] Skipping {skipped} already completed questions; {len(pending_questions)} remaining.")

    new_rows = []
    if pending_questions:
        for start in tqdm(
            range(0, len(pending_questions), args.batch_size),
            desc="Generating Advbench Responses",
        ):
            batch_questions = pending_questions[start : start + args.batch_size]
            batch_responses = generate_batch(batch_questions)
            for question, response in zip(batch_questions, batch_responses):
                new_rows.append({args.question_column: question, "response": response})
    else:
        print("[Resume] No new questions to generate.")

    if existing_df is None:
        df_output = pd.DataFrame(columns=[args.question_column, "response", "response_flag"])
    else:
        df_output = existing_df.copy()
        if "response_flag" not in df_output.columns:
            df_output["response_flag"] = pd.NA

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if "response_flag" not in df_new.columns:
            df_new["response_flag"] = pd.NA
        df_output = pd.concat([df_output, df_new], ignore_index=True)

    if df_output.empty:
        print("No data to save. Exiting.")
        return

    need_moderation_mask = ~df_output["response_flag"].astype(str).isin({"safe", "unsafe"})
    rows_to_moderate = df_output[need_moderation_mask]

    if not rows_to_moderate.empty:
        tokenizer_m, model_m = load_llamaguard()
        for idx, row in tqdm(
            rows_to_moderate.iterrows(),
            total=len(rows_to_moderate),
            desc="Moderating Responses",
        ):
            flag = moderate(row[args.question_column], row["response"], tokenizer_m, model_m)
            df_output.at[idx, "response_flag"] = flag

    total = len(df_output)
    safe_count = df_output["response_flag"].astype(str).str.lower().eq("safe").sum()

    df_output["__order"] = df_output[args.question_column].astype(str).map(normalized_question_indices)
    df_output = df_output.sort_values("__order", kind="stable").drop(columns="__order", errors="ignore")
    df_output.to_csv(args.output, index=False)

    safe_rate = (safe_count / total) * 100 if total > 0 else 0.0
    print(f"Safe@1: {safe_rate:.2f}%")
    print(f"Saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()


