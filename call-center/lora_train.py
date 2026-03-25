from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "./qwen2.5-1.5b" # 或本地模型路径
DATA_PATH = "train.json"

# 1. 读取数据
dataset = load_dataset("json", data_files=DATA_PATH)

# 2. 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    # load_in_8bit=True,
    local_files_only=True,
    device_map="cpu"
)

# 3. 构造对话文本
def format_example(example):
    text = f"用户：{example['instruction']}\n助手：{example['output']}"
    return {"text": text}

dataset = dataset.map(format_example)

# 4. tokenize（不手动造 labels）
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# 5. LoRA 配置
lora_config = LoraConfig(
    r=8, #低秩维度,r 越大，LoRA 能学的东西越多
    lora_alpha=16, #缩放系数,控制 LoRA 更新的“幅度”
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], #LoRA 插在哪些线性层上
    lora_dropout=0.05, #在 LoRA 分支上加 dropout,在 LoRA 分支上加 dropout
    bias="none", #决定是否训练 bias
    task_type="CAUSAL_LM" #告诉 PEFT, 你训练的是：自回归语言模型
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. DataCollator（自动构造 labels）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 7. 训练参数
training_args = TrainingArguments(
    output_dir="./lora-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    report_to="none"
)

# 8. Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    data_collator=data_collator
)

# 9. 开始训练
trainer.train()

# 10. 保存 LoRA 权重
model.save_pretrained("./lora-model")
tokenizer.save_pretrained("./lora-model")
