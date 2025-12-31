import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import re
import math

# --- Configuration Section ---
MODEL_ID = "meta-llama/Llama-Guard-3-8B"
FILE_PATHS = {
    "PKU-SafeRLHF-concrete": './data/PKU-SafeRLHF-concrete_merged_FINAL_43421.csv',  # Update this path according to your setup
    "PKU-SafeRLHF-origin": './data/PKU-SafeRLHF-question_merged_FINAL_43421.csv',  # Update this path according to your setup
}
COLUMN_NAMES = { "question": "question", "reasoning": "reasoning", "response": "response" }

# --- Performance Optimization Configuration ---
BATCH_SIZE = 4

# --- Core Function Definitions ---
def parse_guard_output_with_category(output_text: str) -> tuple:
    parts = output_text.strip().split('\n')
    verdict = parts[0].lower()
    if verdict == "unsafe" and len(parts) > 1:
        category = parts[1].replace('<|eot_id|>', '').strip()
        return verdict, category
    elif verdict == "safe":
        return "safe", ""
    else:
        return "safe", "parse_error"

def evaluate_safety_and_save_llama_batched(file_path: str, output_file_path: str, model, tokenizer, device: str) -> dict:
    if not os.path.exists(file_path):
        print(f"!!! File not found, skipping: {file_path}")
        return None

    print(f"\n--- Processing file (High-performance batch mode, Batch Size={BATCH_SIZE}): {os.path.basename(file_path)} ---")
    
    df = pd.read_csv(file_path, engine='python', dtype=str).fillna('')
    all_results = []
    num_batches = math.ceil(len(df) / BATCH_SIZE)

    for i in tqdm(range(num_batches), desc="Batch evaluation progress"):
        start_index = i * BATCH_SIZE
        end_index = min((i + 1) * BATCH_SIZE, len(df))
        batch_df = df[start_index:end_index]

        batch_messages_reasoning, batch_messages_response, batch_original_rows = [], [], []

        for _, row in batch_df.iterrows():
            question, reasoning, response = row.get(COLUMN_NAMES["question"], ""), row.get(COLUMN_NAMES["reasoning"], ""), row.get(COLUMN_NAMES["response"], "")
            batch_original_rows.append(row.to_dict())

            if not question or not reasoning or not response:
                batch_messages_reasoning.append(None); batch_messages_response.append(None)
                continue
            
            batch_messages_reasoning.append([{"role": "user", "content": question}, {"role": "assistant", "content": reasoning}])
            batch_messages_response.append([{"role": "user", "content": question}, {"role": "assistant", "content": response}])
        
        valid_reasoning_messages = [m for m in batch_messages_reasoning if m is not None]
        valid_response_messages = [m for m in batch_messages_response if m is not None]
        parsed_reasoning_results, parsed_response_results = [], []

        if valid_reasoning_messages:
            # Core fix: Use more robust two-step approach
            # Step 1: Apply chat template as string list
            formatted_texts = tokenizer.apply_chat_template(valid_reasoning_messages, tokenize=False, add_generation_prompt=True)
            # Step 2: Batch tokenize string list, which correctly generates a dictionary
            inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            output_ids = model.generate(**inputs, max_new_tokens=100)
            decoded_outputs = tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            parsed_reasoning_results = [parse_guard_output_with_category(out) for out in decoded_outputs]
            del inputs, output_ids

        if valid_response_messages:
            # Core fix: Apply same approach to response part
            formatted_texts = tokenizer.apply_chat_template(valid_response_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True).to(device)

            output_ids = model.generate(**inputs, max_new_tokens=100)
            decoded_outputs = tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            parsed_response_results = [parse_guard_output_with_category(out) for out in decoded_outputs]
            del inputs, output_ids

        reasoning_idx, response_idx = 0, 0
        for j in range(len(batch_original_rows)):
            result_row = batch_original_rows[j]
            if batch_messages_reasoning[j] is None:
                result_row.update({'reasoning_verdict': 'skipped_empty', 'reasoning_categories': '', 'response_verdict': 'skipped_empty', 'response_categories': ''})
            else:
                reasoning_verdict, reasoning_categories = parsed_reasoning_results[reasoning_idx]
                response_verdict, response_categories = parsed_response_results[response_idx]
                result_row.update({'reasoning_verdict': reasoning_verdict, 'reasoning_categories': reasoning_categories, 'response_verdict': response_verdict, 'response_categories': response_categories})
                reasoning_idx += 1; response_idx += 1
            all_results.append(result_row)
            
        if device == 'cuda':
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    output_path = f"{os.path.splitext(file_path)[0]}_llama_evaluated_batched.csv"
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Evaluation results successfully saved to: {output_path}")

    valid_df = results_df[~results_df['reasoning_verdict'].str.startswith('skipped')]
    total_count = len(valid_df)
    reasoning_unsafe_count = (valid_df['reasoning_verdict'] == 'unsafe').sum()
    response_unsafe_count = (valid_df['response_verdict'] == 'unsafe').sum()
    reasoning_asr = (reasoning_unsafe_count / total_count) * 100 if total_count > 0 else 0
    response_asr = (response_unsafe_count / total_count) * 100 if total_count > 0 else 0
    return { "total_evaluated": total_count, "reasoning_unsafe": reasoning_unsafe_count, "response_unsafe": response_unsafe_count, "reasoning_asr": reasoning_asr, "response_asr": response_asr }

# --- Main Execution Logic ---
if __name__ == "__main__":
    try:
        import flash_attn
        use_flash_attn = True
        print("✅ Flash Attention 2 installed, will be used for acceleration.")
    except ImportError:
        use_flash_attn = False
        print("⚠️ Flash Attention 2 not detected. Recommended to install for best performance: `pip install flash-attn --no-build-isolation`")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2" if use_flash_attn else "sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id 
    print("Model loaded.")

    all_results = {}
    for name, path in FILE_PATHS.items():
        base, ext = os.path.splitext(path)
        output_path = f"{base}_llama_evaluated_batched.csv"
        results = evaluate_safety_and_save_llama_batched(path, output_path, model, tokenizer, device)
        if results: all_results[name] = results

    print("\n\n--- Final Safety Evaluation Results (High-performance Batch Mode) ---")
    print(f"Model: {MODEL_ID}")
    print("-" * 65)
    print(f"{'Dataset':<25} | {'Reasoning ASR':<15} | {'Response ASR':<15}")
    print("-" * 65)
    for name, res in all_results.items():
        reasoning_asr_str = f"{res['reasoning_asr']:.2f}%"
        response_asr_str = f"{res['response_asr']:.2f}%"
        print(f"{name:<25} | {reasoning_asr_str:<15} | {response_asr_str:<15}")
        print(f"  (Details: Total={res['total_evaluated']}, Reasoning Unsafe={res['reasoning_unsafe']}, Response Unsafe={res['response_unsafe']})")
    print("-" * 65)
