# AI-learning

GOAL:
第1周：
Whisper 跑通
Ollama + Qwen API 调通
TTS 能播语音

第2周：
做对话 API
存训练数据
构造 JSON 数据集

第3周：
跑 LoRA
比较微调前后效果

第4周：
整合成语音服务


DAY 1(26.2.9):
语音 → Whisper → Qwen → 语音
- Record audio from microphone  
- Speech-to-text using Whisper  
- Generate response using local Qwen model (Ollama)  
- Text-to-speech using Edge-TTS  
- All local, no cloud dependency

ollama run qwen2.5:7b;
brew install ffmpeg;
pip install openai-whisper edge-tts sounddevice scipy requests

DAY 2(26.2.10):
做对话 API
存训练数据
构造 JSON 数据集

转为LoRA标准格式脚本：convert_dataset.py

微调脚本：lora_train.py (安装依赖 pip install transformers datasets peft accelerate bitsandbytes
)

流程：语音 → dataset.json → LoRA 微调 → 客服模型

Audio → Whisper → Qwen → 保存 → 微调 → 新模型 → 更专业回复

DAY 3(26.2.11):
修改依赖，在本地安装qwen模型
 HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./qwen2.5-1.5b

增加compare脚本，验证微调前后结果

DAY 4(26.2.12):
增加完整脚本voice_service，实现从语音输入到微调到语音输出全流程

