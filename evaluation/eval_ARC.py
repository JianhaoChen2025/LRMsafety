
import argparse
import pandas as pd
from tqdm import tqdm
import os
import ast # Used for safely parsing the choices string
import re # Import the regular expression module
import sys
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Set command-line argument parser
parser = argparse.ArgumentParser(description='Run LLM model on ARC-Challenge-validation data. Processes entire dataset by default if start/end not specified.')
# Change default to None
parser.add_argument('--start', type=int, default=None, help='Start index of the data range (inclusive). If not specified, processes from the beginning.')
# Change default to None
parser.add_argument('--end', type=int, default=None, help='End index of the data range (inclusive). If not specified, processes to the end.')
parser.add_argument('--workers', type=int, default=3, help='Number of concurrent workers (threads).')
parser.add_argument('--model_name', type=str, required=True, help='Name of the language model to use.')
# Keep --dataset for naming the output file, even though input path is fixed
parser.add_argument('--dataset', type=str, default='ARC-Challenge-validation', help='Name of the dataset (used for output file naming).')
parser.add_argument('--output', type=str, default=None, help="Path to save evaluation results. Defaults to 'results/<model>/<dataset>.csv'.")


args = parser.parse_args()

# Get model_name and dataset from command-line arguments
model_name = args.model_name
dataset = args.dataset # Used for output file name
num_workers = args.workers
# Fixed input data path - please update this path according to your setup
input_data_path = './data/ARC-Challenge-validation.csv'

# Ensure results directory exists
model_safe_name = model_name.replace('/', '_')
results_dir = os.path.join('results', model_safe_name)
os.makedirs(results_dir, exist_ok=True)
# Determine output path
output_path = args.output if args.output is not None else os.path.join(results_dir, f'{dataset}.csv')

# Load data
df_data = None
try:
    print(f"Loading data from {input_data_path}")
    if not os.path.exists(input_data_path):
        print(f"Error: Data file not found at '{input_data_path}'.")
        exit(1)
    # Load the entire dataframe
    df_data = pd.read_csv(input_data_path)
    print(f"Data loaded successfully. Total items: {len(df_data)}")

    # Validate essential columns are present
    required_columns = ['id', 'question', 'choices', 'answerKey']
    if not all(col in df_data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df_data.columns]
        print(f"Error: Missing required columns in data file: {missing}")
        exit(1)

except pd.errors.EmptyDataError:
    print(f"Error: Data file at '{input_data_path}' is empty.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading data from '{input_data_path}': {e}")
    exit(1)

# --- Determine the actual processing range ---
data_size = len(df_data)
actual_start = args.start
actual_end = args.end

# If start and end are not specified, process the entire dataset
if actual_start is None and actual_end is None:
    actual_start = 0
    actual_end = data_size - 1
    print(f"No range specified. Processing the entire dataset: [0, {actual_end}]")
elif actual_start is not None and actual_end is None:
     # If only start is specified, process from start to the end
     actual_end = data_size - 1
     print(f"Only start index specified. Processing from index {actual_start} to the end: [{actual_start}, {actual_end}]")
elif actual_start is None and actual_end is not None:
     # If only end is specified, process from the beginning to end
     actual_start = 0
     print(f"Only end index specified. Processing from the beginning to index {actual_end}: [{actual_start}, {actual_end}]")
else:
    # Both start and end are specified, use the provided range
    print(f"Processing specified range: [{actual_start}, {actual_end}]")

# --- Validate the determined range ---
if actual_start < 0 or actual_start >= data_size:
    print(f"Error: Actual start index {actual_start} is out of bounds for dataset size {data_size}.")
    exit(1)
if actual_end < 0 or actual_end >= data_size:
    print(f"Error: Actual end index {actual_end} is out of bounds for dataset size {data_size}.")
    exit(1)
if actual_start > actual_end:
    print(f"Error: Actual start index {actual_start} is greater than actual end index {actual_end}.")
    exit(1)

# Select the data slice for processing based on the actual range
df_slice = df_data.iloc[actual_start:actual_end+1].copy() # Use .copy() to avoid SettingWithCopyWarning
print(f"Target range size: {len(df_slice)} items.")


# --- Resume Capability Enhancement: Read existing results and determine tasks to run ---
completed_ids = set()
if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    print(f"Found existing results file: {output_path}. Reading completed item IDs...")
    try:
        existing_df = pd.read_csv(output_path)
        # Use 'id' for robust resume if available in existing results
        if 'id' in existing_df.columns and not existing_df['id'].empty:
            completed_ids = set(existing_df['id'].dropna().astype(str).tolist())
            print(f"Loaded {len(completed_ids)} previously completed items by ID.")
        # Fallback to checking 'question' if 'id' is not in the results file (e.g., older runs)
        elif 'question' in existing_df.columns and not existing_df['question'].empty:
             # Check if the original data slice contains a 'question' column
            if 'question' in df_slice.columns:
                 completed_questions_in_results = set(existing_df['question'].dropna().astype(str).tolist())
                 # Find IDs from the current slice whose questions are in the completed set
                 completed_ids = set(df_slice[df_slice['question'].isin(completed_questions_in_results)]['id'].dropna().astype(str).tolist())
                 print(f"Loaded {len(completed_ids)} previously completed items by Question (fallback).")
            else:
                 print("Warning: 'id' column not found in existing results and 'question' column not in current data slice. Cannot reliably use question for resume.")
        else:
             print("Existing results file is empty or missing 'id'/'question' columns for resume.")

    except pd.errors.EmptyDataError:
        print("Existing results file is empty after checking size.")
    except Exception as e:
        print(f"Error reading existing results file {output_path} for resume: {e}. Processing all items in the range.")
        completed_ids = set() # Clear completed_ids if reading fails

# Prepare tasks to run (list of row dictionaries)
tasks_to_run: List[Dict[str, object]] = []
# Iterate over the selected slice (df_slice)
for index, row in df_slice.iterrows():
    row_id = str(row['id'])
    # Only add tasks whose ID is not in the completed set
    if row_id not in completed_ids:
        row_dict = row.to_dict()
        row_dict['id'] = row_id
        tasks_to_run.append(row_dict)
    # else:
        # print(f"Skipping ID {row['id']} (already completed).") # Debugging

print(f"Identified {len(tasks_to_run)} new tasks to run within the target range.")

def clean_choices_string(choices_str: str) -> str:
    """Normalize the choices string to simplify parsing."""
    return re.sub(r"array\(\[([^\]]*)\]\,\s*dtype=object\)", r"[\1]", choices_str)


def parse_choices(choices_str: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    cleaned_choices_str = clean_choices_string(choices_str)

    if not cleaned_choices_str.strip().startswith('{') or not cleaned_choices_str.strip().endswith('}'):
        raise ValueError(f"Cleaned choices string is not a valid dictionary format: {cleaned_choices_str}")

    choices_data = ast.literal_eval(cleaned_choices_str)
    choice_texts = choices_data.get('text', [])
    choice_labels = choices_data.get('label', [])

    if not choice_texts or not choice_labels or len(choice_texts) != len(choice_labels):
        raise ValueError("Invalid choices data structure after parsing.")

    return tuple(choice_texts), tuple(choice_labels)


def build_prompt(question: str, choice_texts: Tuple[str, ...], choice_labels: Tuple[str, ...]) -> str:
    prompt_lines = [question.strip(), "", "Choices:"]
    for text, label in zip(choice_texts, choice_labels):
        prompt_lines.append(f"{label}. {text}")
    prompt_lines.append("")
    prompt_lines.append("Please choose the best answer from the choices above. Finish with 'Answer: X' where X is the label of your selected choice.")
    return "\n".join(prompt_lines)


def extract_choice_label(text: Optional[str], valid_labels: Tuple[str, ...]) -> Optional[str]:
    if not text:
        return None

    uppercase_labels = [label.upper() for label in valid_labels]
    label_pattern = "|".join(re.escape(label) for label in uppercase_labels)

    boxed_pattern = re.compile(rf"\\boxed\{{({label_pattern})\}}", flags=re.IGNORECASE)
    boxed_matches = boxed_pattern.findall(text)
    if boxed_matches:
        return boxed_matches[-1].upper()

    answer_pattern = re.compile(
        rf"(?:^|\n)\s*(?:final answer|answer|choice)\s*[:\-]\s*({label_pattern})\b",
        flags=re.IGNORECASE,
    )
    answer_match = answer_pattern.search(text)
    if answer_match:
        return answer_match.group(1).upper()

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        upper_line = stripped.upper()
        if upper_line in uppercase_labels:
            return upper_line
        letter_match = re.search(rf"\b({label_pattern})\b", upper_line)
        if letter_match:
            return letter_match.group(1).upper()

    return None


# --- Model Loading and Inference Helpers (Hugging Face style) ---


def append_results(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    normalized_rows = []
    for row in rows:
        normalized_row = dict(row)
        if 'id' in normalized_row and normalized_row['id'] is not None:
            normalized_row['id'] = str(normalized_row['id'])
        normalized_rows.append(normalized_row)

    df_rows = pd.DataFrame(normalized_rows)
    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    df_rows.to_csv(output_path, mode='a', header=write_header, index=False)

# If no new tasks to run, exit after calculating accuracy based on existing results
if not tasks_to_run:
    print("No new tasks to run in the specified range.")
    # After potentially skipping tasks, still calculate accuracy on the full target range
    # Load the full results file and filter by the original range IDs
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            full_results_df = pd.read_csv(output_path)
            # Filter results to match the IDs in the original range [actual_start, actual_end]
            original_range_ids = set(df_slice['id'].astype(str).tolist())
            results_for_range = full_results_df[full_results_df['id'].astype(str).isin(original_range_ids)].copy() # Use copy() to avoid chain indexing issues later if needed

            if not results_for_range.empty and 'judgment' in results_for_range.columns:
                # Assuming judgment is stored as boolean True/False
                correct_count = results_for_range['judgment'].sum()
                total_count = len(results_for_range) # This is the number of items from the target range that are in the results file
                accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
                print(f"\n--- Accuracy Calculation ---")
                print(f"Range: [{actual_start}, {actual_end}]")
                print(f"Total items in target range: {len(df_slice)}")
                print(f"Items with results in range file (including skipped): {total_count}")
                print(f"Correctly answered: {correct_count}")
                print(f"Accuracy for range [{actual_start}, {actual_end}] based on existing results: {accuracy:.2f}%")
                print(f"----------------------------")
            else:
                print(f"\nNo results found for calculating accuracy in range [{actual_start}, {actual_end}].")

        except Exception as e:
            print(f"\nError calculating accuracy from existing results: {e}")
    else:
        print(f"\nNo results file found at {output_path} to calculate accuracy.")

    exit(0)


# --- Hugging Face Model Inference ---

prepared_items: List[Dict[str, object]] = []
parse_error_rows: List[Dict[str, object]] = []

for item_data in tasks_to_run:
    try:
        choice_texts, choice_labels = parse_choices(item_data['choices'])
        prompt = build_prompt(item_data['question'], choice_texts, choice_labels)
        prepared_items.append(
            {
                'item': item_data,
                'prompt': prompt,
                'choice_labels': choice_labels,
            }
        )
    except Exception as exc:
        error_row = {
            'id': item_data.get('id'),
            'question': item_data.get('question', ''),
            'choices': item_data.get('choices', ''),
            'answerKey': item_data.get('answerKey', ''),
            'llm_answer_raw': f'Error: {exc}',
            'judgment': False,
        }
        parse_error_rows.append(error_row)
        print(f"\nError preparing prompt for ID {item_data.get('id')}: {exc}", file=sys.stderr)

if parse_error_rows:
    append_results(parse_error_rows)

if prepared_items:
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
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
    )

    batch_size = max(1, num_workers)
    progress_bar = tqdm(total=len(prepared_items), desc="Processing data")

    for batch_start in range(0, len(prepared_items), batch_size):
        batch = prepared_items[batch_start:batch_start + batch_size]
        prompts = [entry['prompt'] for entry in batch]

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

        batch_results: List[Dict[str, object]] = []
        for idx, entry in enumerate(batch):
            item_data = entry['item']
            choice_labels = entry['choice_labels']
            generated_sequence = outputs[idx]
            prompt_len = int(prompt_lengths[idx].item())
            generated_tokens = generated_sequence[prompt_len:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            predicted_label = extract_choice_label(output_text, choice_labels)

            answer_key = str(item_data.get('answerKey', '')).strip()
            answer_key_upper = answer_key.upper()
            predicted_normalized = predicted_label.upper() if predicted_label else None
            valid_labels_upper = [label.upper() for label in choice_labels]
            is_valid_prediction = predicted_normalized in valid_labels_upper if predicted_normalized else False

            if predicted_normalized and answer_key_upper:
                judgment = predicted_normalized == answer_key_upper
            else:
                judgment = False

            result_row = {
                'id': item_data.get('id'),
                'question': item_data.get('question'),
                'choices': item_data.get('choices'),
                'answerKey': item_data.get('answerKey'),
                'llm_answer_raw': predicted_normalized if is_valid_prediction else output_text,
                'judgment': judgment,
            }
            batch_results.append(result_row)

        append_results(batch_results)
        progress_bar.update(len(batch_results))

    progress_bar.close()

    print(f"\nProcessing complete for range [{actual_start}, {actual_end}] using model '{model_name}' and dataset '{dataset}'. Results saved incrementally to {output_path}.")
else:
    print("No valid tasks to process after parsing choices.", file=sys.stderr)

# --- Calculate and Print Accuracy ---

# Load the full results file after processing (including newly added items)
if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    try:
        full_results_df = pd.read_csv(output_path)

        # Filter results to match the IDs that were *originally* in the specified range [actual_start, actual_end]
        # Use the df_slice created earlier which contains the data for the target range
        original_range_ids = set(df_slice['id'].astype(str).tolist())
        results_for_range = full_results_df[full_results_df['id'].astype(str).isin(original_range_ids)].copy() # Use copy() to avoid chain indexing issues later if needed

        if not results_for_range.empty and 'judgment' in results_for_range.columns:
            # Assuming judgment is stored as boolean True/False
            correct_count = results_for_range['judgment'].sum() # Sum of True values
            total_count = len(results_for_range) # This is the number of items from the target range that are in the results file

            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0

            print(f"\n--- Accuracy Calculation ---")
            print(f"Range: [{actual_start}, {actual_end}]")
            print(f"Total items in target range: {len(df_slice)}") # Total items originally in the range definition
            print(f"Items with results in range file (including skipped): {total_count}") # Items from the target range found in the results file
            print(f"Correctly answered: {correct_count}")
            print(f"Accuracy for range [{actual_start}, {actual_end}]: {accuracy:.2f}%")
            print(f"----------------------------")

        else:
            print(f"\nNo results found for calculating accuracy for range [{actual_start}, {actual_end}].")

    except Exception as e:
        print(f"\nError calculating accuracy from {output_path}: {e}")
else:
    print(f"\nNo results file found at {output_path} to calculate accuracy.")