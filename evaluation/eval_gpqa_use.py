import argparse
import os
import re
import sys
from typing import Dict, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATA_PATH = "./data/gpqa-use.csv"  # Update this path according to your setup
REQUIRED_COLUMNS = ["Index", "Question", "Option A", "Option B", "Option C", "Option D"]
CHOICE_LABELS = ["A", "B", "C", "D"]
BOXED_PATTERN = re.compile(r"\\boxed\{([A-D])\}", flags=re.IGNORECASE)
ANSWER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:final answer|answer|choice)\s*[:\-]\s*([A-D])\b",
    flags=re.IGNORECASE,
)


def extract_choice(text: Optional[str]) -> Optional[str]:
    """
    Extract a single multiple-choice letter (A-D) from model output.
    Preference order:
      1. Last \\boxed{A-D}
      2. Trailing 'Answer: X' style pattern
      3. Last non-empty line containing an isolated A-D token
    """
    if not text:
        return None

    # 1. Look for boxed answers
    boxed_matches = BOXED_PATTERN.findall(text)
    if boxed_matches:
        return boxed_matches[-1].upper()

    # 2. Look for explicit answer prefixes
    answer_match = ANSWER_PATTERN.search(text)
    if answer_match:
        return answer_match.group(1).upper()

    # 3. Fallback: scan from bottom for standalone choice
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue

        upper_line = stripped.upper()

        # Standalone letter
        if upper_line in CHOICE_LABELS:
            return upper_line

        # First occurrence of any choice letter
        letter_match = re.search(r"\b([A-D])\b", upper_line)
        if letter_match:
            return letter_match.group(1)

    return None


def normalize_choice(value: Optional[str]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None

    letter = str(value).strip().upper()
    return letter if letter in CHOICE_LABELS else None


def load_answer_key(path: Optional[str]) -> Dict[int, str]:
    if not path:
        return {}

    if not os.path.exists(path):
        print(
            f"Warning: answer key '{path}' not found. Continuing without accuracy computation.",
            file=sys.stderr,
        )
        return {}

    try:
        key_df = pd.read_csv(path)
    except Exception as exc:
        print(
            f"Warning: failed to read answer key '{path}' ({exc}). Continuing without accuracy computation.",
            file=sys.stderr,
        )
        return {}

    required_columns = {"Index", "Correct Choice"}
    if not required_columns.issubset(key_df.columns):
        print(
            f"Warning: answer key '{path}' missing required columns {sorted(required_columns)}. "
            "Continuing without accuracy computation.",
            file=sys.stderr,
        )
        return {}

    answer_map: Dict[int, str] = {}
    invalid_entries = 0

    for _, row in key_df.iterrows():
        idx_raw = row.get("Index")
        choice = normalize_choice(row.get("Correct Choice"))

        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            invalid_entries += 1
            continue

        if choice is None:
            invalid_entries += 1
            continue

        answer_map[idx] = choice

    if invalid_entries:
        print(
            f"Warning: answer key '{path}' contained {invalid_entries} invalid entries that were ignored.",
            file=sys.stderr,
        )

    if not answer_map:
        print(
            f"Warning: no valid entries found in answer key '{path}'. Continuing without accuracy computation.",
            file=sys.stderr,
        )

    return answer_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a causal LM on the GPQA USE multiple-choice dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the GPQA USE CSV file (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results. Defaults to 'results/<model>/gpqa-use.csv'.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index (inclusive) of rows to evaluate.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (inclusive) of rows to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--answer-key",
        type=str,
        default=None,
        help="Optional CSV file containing the answer key with columns 'Index' and 'Correct Choice'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of questions to evaluate per generation call.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        print("Error: batch size must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    data_path = args.data
    if not os.path.exists(data_path):
        print(f"Error: dataset file '{data_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        data_df = pd.read_csv(data_path)
    except Exception as exc:
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data_df.columns]
    if missing_columns:
        print(f"Error: dataset missing columns {missing_columns}", file=sys.stderr)
        sys.exit(1)

    # Ensure integer index column
    data_df = data_df[REQUIRED_COLUMNS].copy()

    total_rows = len(data_df)
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else total_rows - 1

    if start_idx < 0 or start_idx >= total_rows:
        print(f"Error: start index {start_idx} out of bounds (0, {total_rows-1}).", file=sys.stderr)
        sys.exit(1)
    if end_idx < 0 or end_idx >= total_rows:
        print(f"Error: end index {end_idx} out of bounds (0, {total_rows-1}).", file=sys.stderr)
        sys.exit(1)
    if start_idx > end_idx:
        print(f"Error: start index {start_idx} greater than end index {end_idx}.", file=sys.stderr)
        sys.exit(1)

    data_slice = data_df.iloc[start_idx : end_idx + 1]

    model_name = args.model
    results_dir = os.path.join("results", model_name.replace("/", "_"))
    os.makedirs(results_dir, exist_ok=True)

    output_path = (
        args.output
        if args.output is not None
        else os.path.join(results_dir, "gpqa-use.csv")
    )

    answer_key_map = load_answer_key(args.answer_key)
    if answer_key_map:
        print(
            f"Loaded {len(answer_key_map)} entries from answer key '{args.answer_key}'.",
            file=sys.stderr,
        )
    elif args.answer_key:
        print(
            "Proceeding without accuracy computation for rows missing in the answer key.",
            file=sys.stderr,
        )

    completed_indices = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            if "Index" in existing_df.columns:
                completed_indices.update(existing_df["Index"].dropna().astype(int).tolist())
                print(
                    f"Found existing results at '{output_path}'. "
                    f"Resuming from {len(completed_indices)} completed rows.",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"Warning: failed to read existing results ({exc}). Recomputing all rows.", file=sys.stderr)

    rows_to_process = [
        record
        for record in data_slice.to_dict(orient="records")
        if record["Index"] not in completed_indices
    ]

    if not rows_to_process:
        print("No new rows to process. Exiting.", file=sys.stderr)
        sys.exit(0)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not provided. Attempting public model download.", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model '{model_name}' on device {device}...", file=sys.stderr)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=hf_token,
        ).to(device)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as exc:
        print(f"Error loading model or tokenizer: {exc}", file=sys.stderr)
        sys.exit(1)

    gen_kwargs = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    newly_completed = []
    missing_answer_indices = set()

    progress_bar = tqdm(total=len(rows_to_process), desc="Evaluating GPQA USE")

    for batch_start in range(0, len(rows_to_process), args.batch_size):
        batch_records = rows_to_process[batch_start : batch_start + args.batch_size]

        prompts = []
        correct_choices = []
        for record in batch_records:
            idx = record["Index"]
            correct_choice = answer_key_map.get(idx)
            if args.answer_key and correct_choice is None:
                missing_answer_indices.add(idx)

            prompt = (
                "Answer the following multiple-choice question.\n"
                "Finish with 'Answer: X' where X is one of A, B, C, or D.\n\n"
                f"Question:\n{record['Question'].strip()}\n\n"
                f"Option A: {record['Option A']}\n"
                f"Option B: {record['Option B']}\n"
                f"Option C: {record['Option C']}\n"
                f"Option D: {record['Option D']}\n"
            )

            prompts.append(prompt)
            correct_choices.append(correct_choice)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            prompt_lengths = torch.full(
                (outputs.size(0),),
                inputs["input_ids"].shape[1],
                dtype=torch.long,
                device=outputs.device,
            )
        else:
            prompt_lengths = attention_mask.sum(dim=1)

        batch_rows = []
        for i, record in enumerate(batch_records):
            idx = record["Index"]
            generated_sequence = outputs[i]
            prompt_len = int(prompt_lengths[i].item())
            generated_tokens = generated_sequence[prompt_len:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            predicted_choice = extract_choice(output_text)
            correct_choice = correct_choices[i]
            is_correct = (
                predicted_choice == correct_choice if correct_choice is not None else None
            )

            batch_rows.append(
                {
                    "Index": idx,
                    "Question": record["Question"],
                    "Option A": record["Option A"],
                    "Option B": record["Option B"],
                    "Option C": record["Option C"],
                    "Option D": record["Option D"],
                    "Correct Choice": correct_choice,
                    "Predicted Choice": predicted_choice,
                    "Is Correct": is_correct,
                    "Prompt": prompts[i],
                    "Model Output": output_text,
                }
            )

        newly_completed.extend(batch_rows)

        df_rows = pd.DataFrame(batch_rows)
        write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
        try:
            df_rows.to_csv(output_path, mode="a", header=write_header, index=False)
        except Exception as exc:
            print(
                f"Error writing results for indices {[row['Index'] for row in batch_rows]}: {exc}",
                file=sys.stderr,
            )

        progress_bar.update(len(batch_records))

    progress_bar.close()

    if missing_answer_indices:
        sample_missing = sorted(list(missing_answer_indices))[:10]
        print(
            f"Warning: answer key missing for {len(missing_answer_indices)} indices. "
            f"Sample missing indices: {sample_missing}",
            file=sys.stderr,
        )

    try:
        final_df = pd.read_csv(output_path)
        print("\n--- Evaluation Summary ---", file=sys.stderr)
        print(f"Total rows evaluated this run: {len(newly_completed)}", file=sys.stderr)
        print(f"Results saved to: {output_path}", file=sys.stderr)

        if "Correct Choice" in final_df.columns and "Predicted Choice" in final_df.columns:
            normalized_correct = final_df["Correct Choice"].apply(normalize_choice)
            normalized_predicted = final_df["Predicted Choice"].apply(normalize_choice)
            valid_mask = normalized_correct.notna()

            if valid_mask.any():
                accuracy = (
                    (normalized_predicted[valid_mask] == normalized_correct[valid_mask])
                    .mean()
                    * 100.0
                )
                print(
                    f"Overall accuracy on rows with answer key entries: {accuracy:.2f}%",
                    file=sys.stderr,
                )
            else:
                print(
                    "Answer key did not provide valid entries for evaluated rows; accuracy not computed.",
                    file=sys.stderr,
                )
        else:
            print(
                "Correct Choice column missing in results; accuracy not computed.",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"Warning: failed to compute summary statistics ({exc}).", file=sys.stderr)


if __name__ == "__main__":
    main()


