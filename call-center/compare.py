from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "./qwen2.5-1.5b"
LORA_PATH = "./lora-model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
lora_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH)

prompt = "我要冻结银行卡"

def ask(model, text):
    inputs = tokenizer(text, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("===== 微调前 =====")
print(ask(base_model, prompt))

print("\n===== 微调后 =====")
print(ask(lora_model, prompt))
