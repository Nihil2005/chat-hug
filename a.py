import os
import logging
from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import torchvision


CACHE_DIR = "E:/cache"
HF_CACHE_DIR = os.path.join(CACHE_DIR, "huggingface")
TORCH_CACHE_DIR = os.path.join(CACHE_DIR, "torch")

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["TORCH_HOME"] = TORCH_CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(TORCH_CACHE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


torchvision.disable_beta_transforms_warning()


torch.hub.set_dir(TORCH_CACHE_DIR)


device = 0 if torch.cuda.is_available() else -1


model_name = "unsloth/Llama-3.2-1B-Instruct"  
config = AutoConfig.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, cache_dir=HF_CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)


llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def chat_with_bot():
    logging.info("Chatbot is ready to interact. Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break


        response = llm(user_input, max_new_tokens=50)  
        print("Chatbot:", response[0]["generated_text"].strip())

if __name__ == "__main__":
    chat_with_bot()
