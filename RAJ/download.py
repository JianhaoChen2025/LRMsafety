import os
import argparse

# Set HF_ENDPOINT if needed (optional, for mirror sites)
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Get HuggingFace token from environment variable
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    raise RuntimeError("Please set HUGGING_FACE_HUB_TOKEN environment variable.")

from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download Llama-Guard-3-8B model")
    parser.add_argument("--local-dir", type=str, default="./models/Llama-Guard-3-8B", 
                        help="Local directory to save the model")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory (optional)")
    args = parser.parse_args()
    
    model = "VincentGOURBIN/Llama-Guard-3-8B"
    snapshot_download(
        repo_id=model,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        max_workers=4,
        cache_dir=args.cache_dir,
        token=hf_token
    )
    print(f"Model downloaded to {args.local_dir}")

if __name__ == "__main__":
    main()
