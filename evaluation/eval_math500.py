import os
import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from concurrent_utils import run_tasks_concurrently

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="math500_eval.csv")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=3, help="Maximum number of concurrent workers.")
    args = parser.parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
    # Update base_url according to your API endpoint
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(base_url=base_url, api_key=openai_api_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto").eval()
    model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{args.n}]")
    def generate(prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
    def judge_math(prompt, response, answer):
        judge_prompt = f"""You are evaluating a math reasoning response.\n\nQuestion:\n{prompt}\n\nModel's Answer:\n{response}\n\nExpected Final Answer:\n{answer}\n\nIs the final numeric answer correct and does the reasoning match?\nOnly reply with: Correct or Incorrect and nothing else. No other extra spaces or words."""
        try:
            result = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=10,
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            return "ERROR"
    dataset_list = list(dataset)

    def process_example(_, row):
        prompt = row["problem"]
        gold = row["answer"]
        try:
            response = generate(prompt)
            label = judge_math(prompt, response, gold)
        except Exception as exc:
            print(f"[WARNING] Failed to evaluate prompt: {exc}")
            response = "ERROR"
            label = "ERROR"
        return {
            "prompt": prompt,
            "response": response,
            "judge_label": label
        }

    rows = [
        r for r in run_tasks_concurrently(
            dataset_list,
            process_example,
            max_workers=args.concurrency,
            desc="Evaluating MATH-500",
        ) if r is not None
    ]
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"✅ Saved to {args.output}")

if __name__ == "__main__":
    main()
