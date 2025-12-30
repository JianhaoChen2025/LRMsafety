"""
Model Configuration File

This file contains configuration for various language models used in the project.
Please update the model_path values according to your local setup:

- For base models: Update paths to point to your local model directories (e.g., ./LLM/ModelName)
- For fine-tuned checkpoints: Update paths to point to your checkpoint directories (e.g., ./checkpoints/ModelName)
- For API models: model_path can be empty, but ensure API keys are set in environment variables

Model types:
- 'api': Model accessed via API (requires API keys in environment variables)
- 'reasoning': Local reasoning model (supports reasoning chains)
- 'instruct': Local instruction-following model
"""

model_cfg = {  
    'qwen-plus-reasoning': {
        'model_type': 'api',
    },
    'R1-32B':{
        'model_path':'./LLM/DeepSeek-R1-Distill-Qwen-32B',  # Update this path to your local model directory
        'model_type':'reasoning'
    },
    'Qwen-32B':{
        'model_path':'./LLM/Qwen2.5-32B-Instruct',  # Update this path to your local model directory
        'model_type':'instruct'
    } ,
    'deepseek-R1':{
        'model_path':'',           # Update this path to your local model directory
        'model_type':'api'
    },
    'deepseek-V3':{
        'model_path':'',           # Update this path to your local model directory
        'model_type':'api'
    },
    'R1-Qwen3-8B':{
        'model_path':'./LLM/DeepSeek-R1-0528-Qwen3-8B',  # Update this path to your local model directory
        'model_type':'reasoning'
    },    
    'Qwen3-8B':{
        'model_path':'./LLM/Qwen3-8B',  # Update this path to your local model directory
        'model_type':'reasoning'
    },          
}