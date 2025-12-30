import os
import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent_utils import run_tasks_concurrently

def extract_final_answer(text):
    import re
    matches = re.findall(r"\[([0-3])\]", text.strip()) 
    
    if matches:
        # Extract the last found answer number (content of capture group (1))
        return int(matches[-1]) 
    else:
        # Fallback: if model didn't use square brackets, try to find a single digit at the end (variant of original logic)
        # Find a single digit 0-3 at the end of text
        match_fallback = re.search(r"([0-3])\s*$", text.strip())
        return int(match_fallback.group(1)) if match_fallback else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="truthfulqa_mc_eval.csv")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum number of concurrent workers.")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file.")

    dataset = load_dataset("EleutherAI/truthful_qa_mc", split="validation", token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_kwargs = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=2048,
        pad_token_id=tokenizer.eos_token_id,
    )

    dataset_list = list(dataset)

    def process_example(_, example):
        question = example["question"]
        choices = example["choices"]
        correct_index = example["label"]
        prompt = (
            f"{question}\n\n"
            + "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            + "\n\nAnswer with the correct option at the end. Here 0 for A, 1 for B, 2 for C and 3 for D. Respond with [0], [1], [2], or [3] only and at the end."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_output = full_response[len(prompt):].strip()
        final_answer = extract_final_answer(model_output)
        is_correct = final_answer == correct_index
        return {
            "question": question,
            "choices": choices,
            "correct_answer": correct_index,
            "full_response": model_output,
            "final_extracted_answer": final_answer,
            "matched": "Yes" if is_correct else "No"
        }

    results = [
        r for r in run_tasks_concurrently(
            dataset_list,
            process_example,
            max_workers=args.concurrency,
            desc="Evaluating TruthfulQA MC",
        ) if r is not None
    ]
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"✅ Saved to → {args.output}")

if __name__ == "__main__":
    main()
