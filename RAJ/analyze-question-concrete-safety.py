import pandas as pd
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import math

# --- 1. Model Loading and Helper Function Definitions ---

# Check if GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Batch size - adjust this value based on GPU memory
BATCH_SIZE = 128 

# Model name
MODEL_NAME = "Qwen/Qwen3Guard-Gen-8B"

print(f"Loading tokenizer for {MODEL_NAME}...")
try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")
    
    print(f"Loading model {MODEL_NAME}...")
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    print("Model loaded successfully.")
    model.eval()
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print("Please ensure you have access to the model and sufficient VRAM if using a GPU.")
    exit()


def extract_label_and_categories(content: str):
    """
    Extract safety labels and categories from model output text.
    """
    if not isinstance(content, str):
        return None, []
        
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
    
    safe_label_match = re.search(safe_pattern, content)
    label = safe_label_match.group(1) if safe_label_match else 'Parse_Error'
    
    categories = re.findall(category_pattern, content)
    
    # If categories are empty but label is not Safe, the model may have only output the label
    if not categories and label != 'Safe':
        content_lower = content.lower()
        if "violent" in content_lower: categories.append("Violent")
        if "illegal" in content_lower: categories.append("Non-violent Illegal Acts")
    
    if "None" in categories:
        categories = ["None"]

    return label, categories


@torch.no_grad()
def get_safety_analysis_batch(prompt_batch: list[str]):
    """
    Use Qwen3Guard model to perform safety analysis on a batch of texts.
    """
    if not prompt_batch:
        return [], []
        
    # Replace empty strings or None with placeholder
    processed_prompts = [p if isinstance(p, str) and p.strip() else "EMPTY_PROMPT" for p in prompt_batch]

    try:
        # Build messages for each prompt in the batch
        batch_messages = [[{"role": "user", "content": p}] for p in processed_prompts]
        
        # Convert messages to model input format text
        batch_text = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in batch_messages]

        # Tokenize and pad the entire batch
        model_inputs = tokenizer(
            batch_text, 
            return_tensors="pt", 
            padding=True,
            truncation=True, 
            max_length=4096
        ).to(model.device)

        # Model generation
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=False
        )
        
        # Remove input portion from generated results
        output_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]

        # Batch decode
        batch_content = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        all_labels = []
        all_categories = []
        
        # Parse results one by one
        for i, content in enumerate(batch_content):
            # If original input is empty, directly return 'Empty'
            if processed_prompts[i] == "EMPTY_PROMPT":
                all_labels.append('Empty')
                all_categories.append([])
                continue
            
            label, categories = extract_label_and_categories(content.strip())
            all_labels.append(label)
            all_categories.append(categories)
            
        return all_labels, all_categories
        
    except Exception as e:
        print(f"An error occurred during model inference batch: {e}")
        error_result = (['Inference_Error'], [[]])
        return [res for res, _ in [error_result] * len(prompt_batch)], [cats for _, cats in [error_result] * len(prompt_batch)]


# --- 2. Main File Processing Logic ---

def process_file():
    """
    Main function to process the specified CSV file and perform safety analysis on question and concrete columns.
    """
    # Target file path - please update this path according to your actual setup
    file_path = "./data/PKU-SafeRLHF.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"\n--- Processing file: {os.path.basename(file_path)} ---")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from the file.")
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return
    
    # Define columns to analyze
    analysis_configs = [
        {'name': 'question', 'source_cols': ['question']},
        {'name': 'concrete', 'source_cols': ['concrete']},
    ]
    
    # Check if required columns exist
    required_cols = ['question', 'concrete']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Analyze each configuration
    for config in analysis_configs:
        name = config['name']
        source_cols = config['source_cols']
        
        if not all(col in df.columns for col in source_cols):
            print(f"Skipping '{name}' as some source columns ({source_cols}) are not in the file.")
            continue
        
        print(f"\nAnalyzing column: '{name}' (from {source_cols})")
        
        # Extract texts to analyze
        combined_texts_series = df.apply(
            lambda row: ' '.join([str(row.get(col) or '') for col in source_cols]).strip(),
            axis=1
        )
        texts_to_check = combined_texts_series.tolist()
        
        all_safety_labels = []
        all_safety_categories = []
        
        # Batch processing loop
        num_batches = math.ceil(len(texts_to_check) / BATCH_SIZE)
        
        # Use tqdm to show batch processing progress
        for i in tqdm(range(0, len(texts_to_check), BATCH_SIZE), desc=f"  Processing {name}", total=num_batches, unit="batch"):
            batch_texts = texts_to_check[i : i + BATCH_SIZE]
            
            # Call batch processing function
            batch_labels, batch_categories = get_safety_analysis_batch(batch_texts)
            
            # Collect results
            all_safety_labels.extend(batch_labels)
            all_safety_categories.extend(batch_categories)
        
        # Add results as new columns to DataFrame
        df[f'{name}_safety_label'] = all_safety_labels
        df[f'{name}_safety_categories'] = [str(cat) if cat else '[]' for cat in all_safety_categories]
        
        print(f"  Completed analysis for '{name}'")
    
    # --- 3. Safety Statistics ---
    print("\n--- Safety Analysis Statistics ---")
    for config in analysis_configs:
        name = config['name']
        label_col_name = f'{name}_safety_label'
        
        if label_col_name in df.columns:
            label_col = df[label_col_name]
            total_valid = label_col.notna().sum()
            if total_valid == 0:
                print(f"  Target '{name}': No valid entries to analyze.")
            else:
                safe_count = (label_col == 'Safe').sum()
                unsafe_count = (label_col == 'Unsafe').sum()
                controversial_count = (label_col == 'Controversial').sum()
                error_count = ((label_col == 'Parse_Error') | (label_col == 'Inference_Error') | (label_col == 'Empty')).sum()
                
                safety_ratio = (safe_count / total_valid) * 100 if total_valid > 0 else 0
                unsafe_ratio = (unsafe_count / total_valid) * 100 if total_valid > 0 else 0
                
                print(f"  Target '{name}':")
                print(f"    Total valid: {total_valid}")
                print(f"    Safe: {safe_count} ({safety_ratio:.2f}%)")
                print(f"    Unsafe: {unsafe_count} ({unsafe_ratio:.2f}%)")
                print(f"    Controversial: {controversial_count}")
                print(f"    Errors: {error_count}")
    
    # --- 4. Save New File ---
    base, ext = os.path.splitext(file_path)
    output_filepath = f"{base}_safety_analyzed{ext}"
    
    try:
        df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        print(f"\nSuccessfully saved results to: {output_filepath}")
        print(f"Total rows saved: {len(df)}")
    except Exception as e:
        print(f"\nError saving file {output_filepath}: {e}")


if __name__ == "__main__":
    print(f"Starting safety analysis with batch size: {BATCH_SIZE}")
    print(f"Model: {MODEL_NAME}")
    process_file()

