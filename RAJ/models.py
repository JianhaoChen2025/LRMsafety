from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from config import model_cfg
import torch
import os

class LLMmodel():
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name not in model_cfg:
            print("Model not supported yet")
        self.model_type = model_cfg[model_name]['model_type']
        if self.model_type != 'api':
            self.model_path = model_cfg[model_name]['model_path']
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True  # May need to add this if loading local custom models
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)  # Load tokenizer using local model path
        else:
            # For API models, may still need tokenizer name (if different)
            if model_name == 'deepseek-R1':
                pass  # This API call doesn't need tokenizer, handled by API
            elif model_name == 'deepseek-V3':
                pass  # This API call doesn't need tokenizer, handled by API
            elif model_name == 'gpt-4o':
                pass  # This API call doesn't need tokenizer, handled by API

    def generate_by_local_model(self, query):
        messages = [
            {"role": "user", "content": query},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=16384
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if self.model_type == 'reasoning':
            tag = "</think>"
            reasoning, content = response.split(tag, 1)
            return reasoning, content
        else:
            return response

    def generate_by_r1(self, query):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Please set DEEPSEEK_API_KEY environment variable.")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        messages = [{"role": "user", "content": query}]
        response = client.chat.completions.create(
            model = "deepseek-reasoner",
            messages = messages,
            temperature = 0.2
        )
        reasoning = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return reasoning, content

    def generate_by_v3(self, query):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Please set DEEPSEEK_API_KEY environment variable.")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query},
            ],
            temperature = 0,
            stream=False
        )
        return response.choices[0].message.content

    def generate_by_gpt(self, query):
        apikey = os.environ.get("OPENAI_API_KEY")
        if not apikey:
            raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
        client = OpenAI(
            api_key = apikey,
        )
        chat_completion = client.chat.completions.create(
            messages = [
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model = "gpt-4o",
            temperature = 0,
        )
        return chat_completion.choices[0].message.content
    

    def generate_by_v3_qwen(self, query):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Please set DASHSCOPE_API_KEY environment variable.")
        client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[              
                {"role": "user", "content": query},
            ],
            temperature = 0,
            stream=False
        )
        return response.choices[0].message.content


    def generate_by_qwen_reasoning(self, query):
        """
        Calls the Qwen model via DashScope API with reasoning mode enabled.
        It streams the response and separates the thinking process from the final answer.
        """
        try:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                raise RuntimeError("Please set DASHSCOPE_API_KEY environment variable.")
            client = OpenAI(
                # It is recommended to use environment variables to manage API keys
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            messages = [{"role": "user", "content": query}]

            completion = client.chat.completions.create(
                model="qwen-plus-2025-04-28",  # Use model that supports thinking functionality
                messages=messages,
                stream=True,  # Must use streaming output to get thinking process
                # extra_body is crucial for passing provider-specific parameters
                extra_body={"enable_thinking": True}, 
            )

            reasoning_content = ""  # Store complete thinking process
            answer_content = ""   # Store complete final answer

            # print("\n" + "=" * 20 + "Thinking process (from Qwen)" + "=" * 20 + "\n")
            is_answering = False

            for chunk in completion:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Collect thinking process content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    # print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

                # Collect final answer content
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        # print("\n\n" + "=" * 20 + "Final Answer (from Qwen)" + "=" * 20 + "\n")
                        is_answering = True
                    # print(delta.content, end="", flush=True)
                    answer_content += delta.content
            
            # print("\n")  # Newline for better output formatting
            return reasoning_content, answer_content

        except Exception as e:
            error_message = f"An error occurred while calling Qwen API: {e}"
            print(error_message)
            return error_message, ""

    def generate(self, query):
        if self.model_type != 'api':
            return self.generate_by_local_model(query)
        else:
            if self.model_name == 'deepseek-R1':
                return self.generate_by_r1(query)
            elif self.model_name == 'deepseek-V3':
                return self.generate_by_v3(query)
            elif self.model_name == 'gpt-4o':
                return self.generate_by_gpt(query)     
            elif self.model_name == 'qwen-plus-reasoning':
                return self.generate_by_qwen_reasoning(query)
            else:
                print(f"No generation method found for API model: {self.model_name}")
                return None, None                