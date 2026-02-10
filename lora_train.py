from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 或本地模型路径
DATA_PATH = "train.json"

dataset = load_dataset("json", data_files=DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto"
)

def format_example(example):
    text = f"用户：{example['instruction']}\n助手：{example['output']}"
    return {"text": text}

dataset = dataset.map(format_example)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./lora-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False
)

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args
)

trainer.train()
model.save_pretrained("./lora-model")
tokenizer.save_pretrained("./lora-model")
