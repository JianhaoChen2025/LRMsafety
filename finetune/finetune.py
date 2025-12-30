import os
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import get_last_checkpoint

def get_columns(dataset_name):
    if dataset_name == "UWNSL/SafeChain":
        return "instruction", "response"
    elif dataset_name == "UCSC-VLAA/STAR-1":
        return "question", "response"
    elif dataset_name.startswith("raj-tomar001/UnSafeChain"):
        return "prompt", "response"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file.")

    print("Loading tokenizer and 8-bit model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=True,
        device_map="auto",
        token=hf_token
    )

    print("Configuring LoRA adapters…")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    print("Loading and tokenizing dataset…")
    if args.subset:
        dataset = load_dataset(args.dataset, name=args.subset, token=hf_token)
    else:
        dataset = load_dataset(args.dataset, token=hf_token)

    col_prompt, col_response = get_columns(args.dataset if not args.subset else f"{args.dataset}/{args.subset}")

    def tokenize_fn(ex):
        text = ex[col_prompt] + "\n" + ex[col_response]
        toks = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
        )
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }

    # First filter out rows where prompt or response is empty
    print("Filtering out rows with None values...")
    filtered_dataset = dataset["train"].filter(
        lambda example: example[col_prompt] is not None and example[col_response] is not None
    )

    print("Tokenizing filtered dataset…")
    tokenized = filtered_dataset.map(
        tokenize_fn,
        batched=False,
        remove_columns=[c for c in filtered_dataset.column_names if c not in [col_prompt, col_response]],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting training…")
    last_checkpoint = get_last_checkpoint(args.output)
    if last_checkpoint is not None:
        print(f"⏮️ Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    print(f"Saving LoRA adapters to {args.output}/adapter_model")
    model.save_pretrained(os.path.join(args.output, "adapter_model"))
    tokenizer.save_pretrained(os.path.join(args.output, "adapter_model"))
    print("Done.")

if __name__ == "__main__":
    main()
