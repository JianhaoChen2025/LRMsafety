# LRMsafety

A comprehensive toolkit for evaluating and analyzing the safety of Large Reasoning Models (LRMs), including jailbreak attacks, safety evaluation, and content analysis.

## 📁 Project Structure

```
LRMsafety/
├── evaluation/          # Safety evaluation scripts for various benchmarks
│   ├── eval_advbench.py
│   ├── eval_ARC.py
│   ├── eval_humaneval.py
│   ├── eval_jailbreakbench.py
│   ├── eval_math500.py
│   ├── eval_strongreject.py
│   ├── eval_truthfulqa_mc.py
│   ├── eval_wildchat.py
│   └── eval_wildjailbreak.py
├── finetune/           # Fine-tuning scripts for safety alignment
│   └── finetune.py
├── RAJ/                # Attack and evaluation tools
│   ├── attack-PKU-SafeRLHF.py
│   ├── analyze-question-concrete-safety.py
│   ├── config.py
│   ├── download.py
│   ├── evaluate-safety-llamaguard.py
│   ├── evaluate-safety-qwen3guard.py
│   ├── gpt5-judge.py
│   └── models.py
├── PGA/                # Content rewriting and safety transformation
│   └── rewrite-1st.py
├── Safety-General-Dilemma/  # Safety dilemma analysis
│   ├── thinking-obey.py
│   ├── thinking-refuse.py
│   └── thinking-way-bingfa.py
├── analysis/           # Data analysis and visualization
│   ├── combine_differential_wordclouds.py
│   ├── generate_differential_wordclouds.py
│   └── question-length.py
└── data/               # Data files (update paths as needed)
```

## 🚀 Quick Start

### Prerequisites

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch transformers pandas tqdm openai datasets peft accelerate
pip install matplotlib wordcloud  # For analysis scripts
```

### Environment Variables

Set the following environment variables before running the code:

```bash
# API Keys (required for API-based models)
export DASHSCOPE_API_KEY="your-dashscope-api-key"      # For Qwen models
export DEEPSEEK_API_KEY="your-deepseek-api-key"        # For DeepSeek models
export OPENAI_API_KEY="your-openai-api-key"            # For OpenAI models
export OPENAI_BASE_URL="your-api-base-url"             # Optional, defaults to https://api.openai.com/v1

# HuggingFace Tokens (required for private datasets/models)
export HF_TOKEN="your-huggingface-token"
export HUGGING_FACE_HUB_TOKEN="your-huggingface-token"
```

### Configuration

1. **Model Configuration**: Update `RAJ/config.py` with your local model paths:
   - Base models: `./LLM/ModelName`
   - Fine-tuned checkpoints: `./checkpoints/ModelName`
   - API models: Leave `model_path` empty, set API keys in environment variables

2. **Data Paths**: Update file paths in scripts according to your setup:
   - Input data: `./data/`
   - Output results: `./output/` or `./results/`

## 📊 Datasets

### Fine-tuning Datasets

Fine-tuning datasets are available at: **chenjacike220/LRMsafety**

To use these datasets:

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("chenjacike220/LRMsafety", token=os.environ.get("HF_TOKEN"))
```

Supported datasets include:
- UWNSL/SafeChain
- UCSC-VLAA/STAR-1
- raj-tomar001/UnSafeChain
- PGA-1000/2000/3000/Full

## 🔧 Usage Examples

### 1. Fine-tuning a Model

```bash
python finetune/finetune.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --dataset "UWNSL/SafeChain" \
    --output "./checkpoints/my-model" \
    --epochs 2 \
    --lr 1e-5 \
    --batch 1 \
    --grad_accum 8
```

### 2. Evaluating Model Safety

```bash
# Evaluate on AdvBench
python evaluation/eval_advbench.py \
    --model "your-model-path" \
    --input "./data/Advbench.csv" \
    --output "./results/advbench_eval.csv"

# Evaluate on JailbreakBench
python evaluation/eval_jailbreakbench.py \
    --model "your-model-path" \
    --output "./results/jailbreakbench_eval.csv"
```

### 3. Running Attack Experiments

```bash
python RAJ/attack-PKU-SafeRLHF.py \
    --model_name "R1-32B" \
    --dataset_name "PKU-SafeRLHF-prompts" \
    --start_id 0 \
    --end_id 100
```

### 4. Safety Evaluation

```bash
# Using Qwen3Guard
python RAJ/evaluate-safety-qwen3guard.py

# Using LlamaGuard
python RAJ/evaluate-safety-llamaguard.py
```

### 5. Content Rewriting

```bash
python PGA/rewrite-1st.py
# Update INPUT_FILE_PATH and FINAL_OUTPUT_PATH in the script
```

## 📈 Analysis Tools

### Word Cloud Generation

```bash
# Generate differential word clouds
python analysis/generate_differential_wordclouds.py

# Combine word clouds
python analysis/combine_differential_wordclouds.py
```

### Length Analysis

```bash
python analysis/question-length.py
```

## 🎯 Evaluation Benchmarks

This project supports evaluation on multiple safety benchmarks:

- **AdvBench**: Adversarial prompts for safety evaluation
- **JailbreakBench**: Comprehensive jailbreak evaluation
- **WildChat**: Real-world user prompts
- **WildJailbreak**: Jailbreak attempts from real conversations
- **StrongREJECT**: Strong refusal evaluation
- **TruthfulQA**: Truthfulness evaluation
- **HumanEval**: Code generation safety
- **ARC**: AI2 Reasoning Challenge
- **MATH-500**: Mathematical reasoning

## 🔐 Security Notes

- All API keys should be stored in environment variables, never hardcoded
- Update all file paths in scripts to match your local setup
- Review and adjust model paths in `RAJ/config.py` before use

## 🙏 Acknowledgments

Parts of this codebase are derived from [UnsafeChain](https://github.com/mbzuai-nlp/UnsafeChain.git),  which designed to fine-tune large reasoning models to be safer and more factual. We thank the authors for their valuable contributions to the field of reasoning model safety.

## 📄 License

[Apache-2.0 license]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

[chenjianhao@whu.edu.cn]

