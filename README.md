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


DAY 1:
语音 → Whisper → Qwen → 语音
- 🎙 Record audio from microphone  
- 🧠 Speech-to-text using Whisper  
- 🤖 Generate response using local Qwen model (Ollama)  
- 🔊 Text-to-speech using Edge-TTS  
- 💻 Fully local, no cloud dependency

ollama run qwen2.5:7b
brew install ffmpeg
pip install openai-whisper edge-tts sounddevice scipy requests
