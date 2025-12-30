import os
import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import sys  # Import sys module for error logging
import time

from concurrent_utils import run_tasks_concurrently

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID or path.")
    parser.add_argument("--output", type=str, default="humaneval_eval.csv", help="Output CSV file name.")
    parser.add_argument("--n", type=int, default=200, help="Number of samples to evaluate.")
    parser.add_argument("--concurrency", type=int, default=3, help="Maximum number of concurrent evaluation threads.")
    args = parser.parse_args()

    # Use qwen-plus as the judge model
    # API Key must be obtained from environment variable
    openai_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Please set DASHSCOPE_API_KEY environment variable.")
    
    # Initialize OpenAI client for calling qwen-plus judge service
    try:
        client = OpenAI(
            api_key=openai_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60.0  # Set default timeout to 60 seconds
        )
        print("Initialized qwen-plus client for code evaluation.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

    # Initialize local model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
            device_map="auto"  # Automatically allocate devices, helpful for large models
        ).eval()
        model.to(device)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        sys.exit(1)

    # Load HumanEval dataset
    try:
        # First load full dataset to check actual size
        ds_full = load_dataset("openai/openai_humaneval", split="test")
        actual_size = len(ds_full)
        print(f"HumanEval dataset actual size: {actual_size} examples")
        
        # If requested number exceeds actual size, use actual size
        if args.n > actual_size:
            print(f"Warning: Requested {args.n} samples but dataset only has {actual_size}. Using all {actual_size} samples.")
            args.n = actual_size
        
        ds = load_dataset("openai/openai_humaneval", split=f"test[:{args.n}]")
        print(f"Loaded {len(ds)} examples for evaluation.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


    def generate(prompt):
        """Local model code generation function"""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Add try-except to catch OOM and other generation errors
        try:
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=2048, 
                pad_token_id=tokenizer.eos_token_id, 
                do_sample=False  # Default to greedy search
            )
            
            # Since input_ids has been passed in, output_ids[0] contains both prompt and generated code
            full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Safely remove prompt portion
            if full_output.startswith(prompt):
                 return full_output[len(prompt):].strip()
            else:
                 # Some models may behave differently, directly return generated text
                 return full_output.strip()

        except torch.cuda.OutOfMemoryError as oom_e:
            # Catch most common OOM error
            print(f"\n[ERROR] CUDA Out of Memory during generation for prompt length {len(input_ids[0])}. Skipping.")
            # Try to clear cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            raise oom_e  # Re-raise to let outer layer catch and record
        except Exception as e:
            # Catch other generation errors
            print(f"\n[ERROR] Unexpected error during generation: {e}")
            raise e


    def judge_code(prompt, response, reference, max_retries=3, timeout=60):
        """External API code judgment function with retry and timeout mechanism"""
        judge_prompt = f"""You are a Python code evaluator. The task is to check if the Model's Code is functionally and semantically equivalent to the Reference Solution for the given Prompt.
\n\nPrompt:\n{prompt}\n\nModel's Code:\n{response}\n\nReference Solution:\n{reference}\n\nIs the model's response functionally and semantically equivalent to the reference solution? Just check if the solution is logically correct not the documentations, etc. Respond with just \"correct\" or \"incorrect\" and nothing else."""
        
        start_time = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                result = client.chat.completions.create(
                    model="qwen-plus",  # Use qwen-plus as judge model
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0,
                    max_tokens=10,
                )
                elapsed = time.time() - start_time
                
                # Ensure returned content is lowercase and cleaned
                label = result.choices[0].message.content.strip().lower()
                
                # Validate if returned label is valid
                if label not in ['correct', 'incorrect']:
                    print(f"\n[WARNING] Invalid label '{label}' returned, treating as 'incorrect'")
                    label = 'incorrect'
                
                return label
            
            except Exception as e:
                error_msg = str(e)
                elapsed = time.time() - start_time if start_time is not None else 0
                
                # If this is the last attempt, raise exception
                if attempt == max_retries - 1:
                    print(f"\n[ERROR] API Call Failed after {max_retries} attempts (last attempt took {elapsed:.2f}s): {error_msg}")
                    raise e
                else:
                    # Wait before retry
                    wait_time = (attempt + 1) * 2  # Incremental wait time: 2s, 4s, 6s
                    print(f"\n[WARNING] API Call Failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {error_msg}")
                    time.sleep(wait_time)
                    start_time = None  # Reset for next attempt
            
            
    ds_list = list(ds)
    expected_count = len(ds_list)
    print(f"Expected to process {expected_count} examples.")

    def process_example(i, row):
        """Process a single example, ensuring result is always returned"""
        task_id = row.get("task_id", f"task_{i}")
        prompt = row["prompt"]
        answer = row["canonical_solution"]
        
        # Add instruction prefix to limit model to only generate code
        instruction = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"
        full_prompt = instruction + prompt
        
        try:
            # Generate code
            response = generate(full_prompt)
            
            # Judge code
            label = judge_code(prompt, response, answer)
            
            return {
                "task_id": task_id,
                "prompt": prompt,
                "response": response,
                "judge_label": label
            }
        except Exception as e:
            error_msg = str(e).split('\n')[0]
            print(f"\n[FATAL SKIP] Task {i} (ID: {task_id}) FAILED! Error: {error_msg}")
            print("Recording as 'CRASHED'.")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Ensure result is always returned, even on failure
            return {
                "task_id": task_id,
                "prompt": prompt,
                "response": "ERROR_CRASHED",
                "judge_label": f"CRASHED: {error_msg}"
            }

    print(f"\nStarting evaluation with {args.concurrency} concurrent workers...")
    results = run_tasks_concurrently(
        ds_list,
        process_example,
        max_workers=args.concurrency,
        desc="Evaluating HumanEval",
    )
    
    # Filter out None results (theoretically shouldn't exist since process_example always returns result)
    rows = [r for r in results if r is not None]
    
    # Check if there are missing results
    if len(rows) < expected_count:
        print(f"\n[WARNING] Expected {expected_count} results but got {len(rows)}. Some tasks may have failed silently.")
        # Find missing task_ids
        completed_task_ids = {r['task_id'] for r in rows}
        all_task_ids = {row.get("task_id", f"task_{i}") for i, row in enumerate(ds_list)}
        missing_task_ids = all_task_ids - completed_task_ids
        if missing_task_ids:
            print(f"Missing task IDs: {sorted(missing_task_ids)}")
    
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    
    # Statistics
    success_count = len(df[~df['response'].str.contains('ERROR_CRASHED', na=False)])
    crashed_count = len(df[df['response'].str.contains('ERROR_CRASHED', na=False)])
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Total processed: {len(df)}/{expected_count}")
    print(f"   Successful: {success_count} ({success_count/len(df)*100:.1f}%)")
    print(f"   Crashed: {crashed_count} ({crashed_count/len(df)*100:.1f}%)")
    print(f"   Results saved to → {args.output}")
    
    # Calculate and print accuracy
    # Filter out valid judgment labels (excluding CRASHED and other error cases)
    valid_labels = df[df['judge_label'].notna() & (~df['judge_label'].str.contains('CRASHED', na=False, regex=True))]
    
    if len(valid_labels) > 0:
        correct_count = len(valid_labels[valid_labels['judge_label'].str.lower() == 'correct'])
        incorrect_count = len(valid_labels[valid_labels['judge_label'].str.lower() == 'incorrect'])
        total_valid = len(valid_labels)
        accuracy = (correct_count / total_valid * 100) if total_valid > 0 else 0.0
        
        print(f"\n📊 Accuracy Statistics:")
        print(f"   Valid evaluations: {total_valid}/{len(df)}")
        print(f"   Correct: {correct_count} ({correct_count/total_valid*100:.2f}%)")
        print(f"   Incorrect: {incorrect_count} ({incorrect_count/total_valid*100:.2f}%)")
        print(f"   ⭐ Accuracy: {accuracy:.2f}%")
    else:
        print(f"\n⚠️  No valid judge labels found for accuracy calculation.")

if __name__ == "__main__":
    main()