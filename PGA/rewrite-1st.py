# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import concurrent.futures

# ==============================================================================
# --- 1. Comprehensive Configuration Section ---
# ==============================================================================

# --- Step 1: Content Safety Rewriting Configuration ---
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("Please set DASHSCOPE_API_KEY environment variable.")
client = OpenAI(
    api_key=api_key, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
REWRITER_MODEL_NAME = "qwen-plus"
# Set maximum generation tokens to ensure sufficient space to generate rewritten content similar in length to original
MAX_TOKENS = 18000

# --- Step 2: Safety Evaluation Configuration ---
EVALUATOR_MODEL_ID = "meta-llama/Llama-Guard-3-8B"

# --- File Paths and Test Configuration ---
# Please update these paths according to your setup
INPUT_FILE_PATH = './data/problematic_samples_1203.csv'
FINAL_OUTPUT_PATH = './output/problematic_samples_1203-rewrite-3.csv'

# Set to None to process entire file; for quick testing, specify row count via TEST_ROW_COUNT environment variable
_TEST_ROW_COUNT_ENV = os.getenv("TEST_ROW_COUNT")
TEST_ROW_COUNT = (
    int(_TEST_ROW_COUNT_ENV)
    if _TEST_ROW_COUNT_ENV not in (None, "", "None")
    else None
)

# --- Performance and Batch Processing Configuration ---
BATCH_SIZE = 4 
MAX_PARALLEL_WORKERS = 10

# --- Prompt Templates (Updated) ---
SYSTEM_PROMPT = """
You are an expert in AI ethics and content safety. Your task is to evaluate and rewrite the given text to eliminate potentially harmful, unethical, or dangerous content.

**CRITICAL REQUIREMENT: LENGTH PRESERVATION**
You MUST maintain the length of the rewritten content to be approximately 80%-120% of the original content length (measured in characters). This is a mandatory requirement that is equally important as safety. Do NOT significantly shorten the content. If the original reasoning is 4000 characters, your rewritten reasoning must be around 3200-4800 characters. If the original response is 2000 characters, your rewritten response must be around 1600-2400 characters.

**ESPECIALLY CRITICAL FOR REASONING:**
The reasoning section is particularly important. You MUST maintain the same level of detail, the same number of reasoning steps, and similar elaboration in the rewritten reasoning. If the original reasoning has 10 reasoning steps, your rewritten reasoning should also have approximately 8-12 steps. If the original reasoning explores multiple aspects or perspectives, your rewritten reasoning should explore a similar number of safe alternatives. DO NOT condense or summarize the reasoning - expand on safe alternatives to maintain length.

**CRITICAL FOR RESPONSE: LENGTH CONTROL**
The response section must NOT exceed 120% of the original response length. If the original response is 2000 characters, your rewritten response must be at most 2400 characters. Target 85%-115% to ensure you stay well within the requirement. If your draft exceeds 120%, you MUST trim it down by removing redundant explanations, consolidating points, or being more concise. Response should be concise but complete - do not over-elaborate.

You must strictly adhere to one or more of the five strategies provided below and explicitly state which strategy (or strategies) you used in your response.
"""


USER_PROMPT_TEMPLATE = """
**Core Task:**
For the `[Input Question]` provided below, perform a safety rewrite on both the `[Original Reasoning]` and the `[Original Response]`. After rewriting, you must explicitly state which specific strategy or strategies you used for each part (reasoning and response).

**WHAT TO DO:**
- Write as if you are directly answering the user's question with safe, helpful content
- Use phrases like "Instead, we should..." or "A better approach is..." (referring to safe alternatives, NOT comparing to original)
- Present your reasoning and response as if they are the first and only response to the question
- Focus on safe, constructive content without any meta-commentary about what was removed or changed
- The Rewritten Result should be pure content only - no comments, no notes, no verification sections

**VIOLATION CHECK:** Before submitting, scan your entire output (including any notes or comments) for any mention of "original", "provided", or comparison phrases. If found anywhere, remove or rewrite that section to be completely independent.

**CRITICAL LENGTH REQUIREMENT:**
**You MUST maintain similar length to the original content. This is a mandatory requirement.**
- The rewritten reasoning should be approximately **80%-120%** of the original reasoning length (count characters, not words).
- The rewritten response should be approximately **80%-120%** of the original response length (count characters, not words).
- **DO NOT significantly shorten the content.** If the original reasoning is 4000 characters, your rewritten reasoning should be around 3200-4800 characters.
- **DO NOT significantly lengthen the content.** Keep it within the specified range.
- **This length requirement is as important as the safety requirement.** You must balance both safety and length.
- **IMPORTANT:** Even when removing harmful content, you must maintain the same level of detail and elaboration. Replace harmful details with safe, educational, or constructive content of similar length. Do not simply delete harmful parts - expand on safe alternatives to maintain length.

**Key Principle: Balance Safety with Helpfulness**
While safety is the absolute priority, the rewritten content should remain a relevant and helpful response to the user's query. Avoid generic, unhelpful refusals. The goal is to transform the original harmful response into a new one that is both **safe and useful** for the user, addressing their underlying intent whenever possible.
You should rewrite the [Original Reasoning] to start with **Okay, so **.

**CRITICAL REMINDER FOR REASONING:**
The reasoning section is your thinking process. It should be as detailed and thorough as the original reasoning, but exploring safe alternatives instead of harmful methods. If the original reasoning thinks through 8 different aspects, your rewritten reasoning should think through approximately 8 safe aspects. If the original reasoning elaborates on each point with 200 characters, your rewritten reasoning should elaborate on safe points with similar length. **Length preservation in reasoning is non-negotiable.**

**Five Rewriting Strategy Definitions:**
1.  **Goal Reframing:** Ignore the specific method in the user's request, and instead focus on and satisfy their underlying intent. You need to perform a layer of inference to deduce the user's true goal from their direct request, and then provide a safer, more effective, or even more professional solution.
2.  **Information Downgrading:** Provide 'what' and 'why' knowledge, but deliberately omit or obscure the key 'how-to' details to reduce the actionability of the information.
3.  **Counterfactual & Risk Clarification:** Do not directly answer the user's question. Instead, reveal the significant, hidden risks behind the question that the user may not be aware of, and explain the severe consequences of 'what would happen if you did that'.
4.  **Premise Rejection & Positive Reframing:** First, clearly identify and reject the incorrect, dangerous, or unethical premise in the user's question. Then, reframe the question into a more positive and constructive discussion.
5.  **Empathetic Redirection & Resource Guidance:** When you identify that a user may be in distress, struggling, or in a dangerous situation, abandon any discussion of the question's content itself. Shift the entire focus to expressing empathy, validating their feelings, and providing professional, immediately actionable resources for help.

**Data to Process:**

[Input Question]: 
{question}

[Original Reasoning]:
{reasoning}

[Original Response]:
{response}

**Output Format:**

### Reasoning Rewrite

* **Strategy Used:**
    <Clearly list one or more strategy names here>

* **Rewritten Result:**
    <Present the new rewritten reasoning process here. 
    
### Response Rewrite

* **Strategy Used:**
    <Clearly list one or more strategy names here>

* **Rewritten Result:**
    <Present the new, user-facing rewritten response here.
    
"""


# ==============================================================================
# --- 2. Core Function Definitions ---
# ==============================================================================

def _parse_llm_output(raw_output: str) -> dict:
    """
    Parse LLM output, compatible with Markdown style returned by deepseek.
    Use regex to locate titles and fields, avoiding split failures due to line break or space differences.
    """
    parsed = {
        'reasoning_strategy': "Parsing Error",
        'rewritten_reasoning': "Parsing Error",
        'response_strategy': "Parsing Error",
        'rewritten_response': "Parsing Error",
    }
    if not isinstance(raw_output, str) or not raw_output.strip():
        return parsed

    reasoning_pattern = re.compile(
        r"###\s*Reasoning\s*Rewrite.*?\*\s*\*\*Strategy Used:\*\*\s*(.*?)\s*\*\s*\*\*Rewritten Result:\*\*\s*(.*?)(?=###\s*Response\s*Rewrite|$)",
        re.S | re.I,
    )
    response_pattern = re.compile(
        r"###\s*Response\s*Rewrite.*?\*\s*\*\*Strategy Used:\*\*\s*(.*?)\s*\*\s*\*\*Rewritten Result:\*\*\s*(.*)",
        re.S | re.I,
    )

    reasoning_match = reasoning_pattern.search(raw_output)
    if reasoning_match:
        parsed['reasoning_strategy'] = reasoning_match.group(1).strip()
        parsed['rewritten_reasoning'] = reasoning_match.group(2).strip()

    response_match = response_pattern.search(raw_output)
    if response_match:
        parsed['response_strategy'] = response_match.group(1).strip()
        parsed['rewritten_response'] = response_match.group(2).strip()

    return parsed

def process_single_row(index, row):
    """
    Process rewriting task for a single row of data, designed for parallel invocation.
    """
    try:
        user_content = USER_PROMPT_TEMPLATE.format(
            question=row['question'],
            reasoning=row['reasoning'],
            response=row['response']
        )
        completion = client.chat.completions.create(
            model=REWRITER_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.5,
            stream=False,
        )
        raw_output = completion.choices[0].message.content
        parsed_output = _parse_llm_output(raw_output)
        return {'index': index, 'llm_raw_output': raw_output, **parsed_output}
    except Exception as e:
        return {
            'index': index,
            'llm_raw_output': f"Error: {e}",
            'rewritten_reasoning': "Processing Error",
            'reasoning_strategy': "Processing Error",
            'rewritten_response': "Processing Error",
            'response_strategy': "Processing Error",
        }

def rewrite_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use ThreadPoolExecutor to process data rewriting in parallel.
    """
    print(f"\n--- Step 1: Starting Content Safety Rewriting (using {REWRITER_MODEL_NAME}, parallel workers: {MAX_PARALLEL_WORKERS}) ---")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        future_to_row = {executor.submit(process_single_row, index, row): index for index, row in df.iterrows()}
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(df), desc="Safety rewriting progress"):
            results.append(future.result())
    results_df = pd.DataFrame(results).set_index('index').sort_index()
    overlapping_cols = set(df.columns) & set(results_df.columns)
    if overlapping_cols:
        df = df.drop(columns=list(overlapping_cols))
    df = df.join(results_df)
    print("✅ Content safety rewriting completed.")
    return df

def parse_guard_output_with_category(output_text: str) -> tuple:
    parts = output_text.strip().split('\n')
    verdict = parts[0].lower()
    if verdict == "unsafe" and len(parts) > 1:
        category = parts[1].replace('<|eot_id|>', '').strip()
        return "unsafe", category
    elif verdict == "safe":
        return "safe", ""
    else:
        return "safe", "parse_error"

def evaluate_rewritten_data(df: pd.DataFrame, model, tokenizer, device: str) -> pd.DataFrame:
    print(f"\n--- Step 2: Starting Safety Evaluation (using {EVALUATOR_MODEL_ID}, Batch Size={BATCH_SIZE}) ---")
    all_results = []
    num_batches = math.ceil(len(df) / BATCH_SIZE)
    for i in tqdm(range(num_batches), desc="Safety evaluation progress"):
        start_index = i * BATCH_SIZE
        end_index = min((i + 1) * BATCH_SIZE, len(df))
        batch_df = df.iloc[start_index:end_index]
        batch_messages_reasoning, batch_messages_response, batch_original_rows = [], [], []
        for _, row in batch_df.iterrows():
            question = row.get('question', '')
            rewritten_reasoning = row.get('rewritten_reasoning', '')
            rewritten_response = row.get('rewritten_response', '')
            batch_original_rows.append(row.to_dict())
            if rewritten_reasoning in ["", "Parsing Error", "Processing Error"]:
                 batch_messages_reasoning.append(None)
            else:
                batch_messages_reasoning.append([{"role": "user", "content": question}, {"role": "assistant", "content": rewritten_reasoning}])
            if rewritten_response in ["", "Parsing Error", "Processing Error"]:
                batch_messages_response.append(None)
            else:
                batch_messages_response.append([{"role": "user", "content": question}, {"role": "assistant", "content": rewritten_response}])

        valid_reasoning_messages = [m for m in batch_messages_reasoning if m is not None]
        valid_response_messages = [m for m in batch_messages_response if m is not None]
        parsed_reasoning_results, parsed_response_results = [], []

        if valid_reasoning_messages:
            formatted_texts = tokenizer.apply_chat_template(valid_reasoning_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            output_ids = model.generate(**inputs, max_new_tokens=100)
            decoded_outputs = tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            parsed_reasoning_results = [parse_guard_output_with_category(out) for out in decoded_outputs]
            del inputs, output_ids

        if valid_response_messages:
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
                result_row.update({'guard_reasoning_verdict': 'skipped_empty_or_error', 'guard_reasoning_categories': ''})
            else:
                verdict, category = parsed_reasoning_results[reasoning_idx]
                result_row.update({'guard_reasoning_verdict': verdict, 'guard_reasoning_categories': category})
                reasoning_idx += 1
            if batch_messages_response[j] is None:
                result_row.update({'guard_response_verdict': 'skipped_empty_or_error', 'guard_response_categories': ''})
            else:
                verdict, category = parsed_response_results[response_idx]
                result_row.update({'guard_response_verdict': verdict, 'guard_response_categories': category})
                response_idx += 1
            all_results.append(result_row)
        if device == 'cuda':
            torch.cuda.empty_cache()
    print("✅ Safety evaluation completed.")
    return pd.DataFrame(all_results)


# ==============================================================================
# --- 3. Main Execution Logic ---
# ==============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Will use 'sdpa' as attention mechanism.")

    print(f"Loading safety evaluation model: {EVALUATOR_MODEL_ID}...")
    evaluator_model = AutoModelForCausalLM.from_pretrained(
        EVALUATOR_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa"
    )
    evaluator_tokenizer = AutoTokenizer.from_pretrained(EVALUATOR_MODEL_ID)
    evaluator_tokenizer.pad_token = evaluator_tokenizer.eos_token
    evaluator_model.config.pad_token_id = evaluator_tokenizer.eos_token_id 
    print("Safety evaluation model loaded.")

    try:
        df = pd.read_csv(INPUT_FILE_PATH, nrows=TEST_ROW_COUNT)
        if TEST_ROW_COUNT is not None:
            print(f"\nSuccessfully loaded first {len(df)} rows from {INPUT_FILE_PATH} for testing.")
        else:
            print(f"\nSuccessfully loaded {len(df)} complete rows from {INPUT_FILE_PATH}.")
    except FileNotFoundError:
        print(f"!!! Error: Input file not found, please check path: {INPUT_FILE_PATH}")
        return
    except Exception as e:
        print(f"!!! Error: Error occurred while loading CSV file: {e}")
        return

    df_rewritten = rewrite_data(df)
    df_final = evaluate_rewritten_data(df_rewritten, evaluator_model, evaluator_tokenizer, device)

    print(f"\n--- Step 3: Saving Final Results ---")
    df_final.to_csv(FINAL_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ Complete processing results successfully saved to: {FINAL_OUTPUT_PATH}")

    print("\n\n--- Safety Rewriting Effectiveness Evaluation Report ---")
    print(f"Rewriting model: {REWRITER_MODEL_NAME}")
    print(f"Evaluation model: {EVALUATOR_MODEL_ID}")
    print("-" * 70)
    
    valid_df = df_final[~df_final['guard_reasoning_verdict'].str.startswith('skipped')]
    total_count = len(valid_df)
    
    if total_count > 0:
        reasoning_unsafe_count = (valid_df['guard_reasoning_verdict'] == 'unsafe').sum()
        response_unsafe_count = (valid_df['guard_response_verdict'] == 'unsafe').sum()
        reasoning_unsafe_rate = (reasoning_unsafe_count / total_count) * 100
        response_unsafe_rate = (response_unsafe_count / total_count) * 100
        
        print(f"{'Evaluation Item':<25} | {'Unsafe Rate (lower is better)':<25} | {'Details'}")
        print("-" * 70)
        
        reasoning_rate_str = f"{reasoning_unsafe_rate:.2f}%"
        response_rate_str = f"{response_unsafe_rate:.2f}%"
        
        print(f"{'Rewritten Reasoning':<25} | {reasoning_rate_str:<25} | {reasoning_unsafe_count} / {total_count} still unsafe")
        print(f"{'Rewritten Response':<25} | {response_rate_str:<25} | {response_unsafe_count} / {total_count} still unsafe")
    else:
        print("No valid data available for evaluation.")
    print("-" * 70)

if __name__ == "__main__":
    main()
