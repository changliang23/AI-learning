
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import asyncio
import edge_tts
import os
import json
from datetime import datetime

AUDIO_FILE = "input.wav"
REPLY_FILE = "reply.mp3"
DATASET_FILE = "dataset.json"

# 1. 录音
def record_audio(duration=5, fs=16000):
    print("开始录音...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(AUDIO_FILE, fs, audio)
    print("录音完成")

# 2. Whisper 转文字
def speech_to_text():
    print("Whisper 转文字...")
    model = whisper.load_model("base")
    result = model.transcribe(AUDIO_FILE, language="zh")
    print("识别结果:", result["text"])
    return result["text"]

# 3. 调用本地 Ollama(Qwen)
def call_qwen(prompt):
    print("调用 Qwen...")
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(url, json=data)
    return r.json()["response"]

# 4. TTS（生成 MP3）
async def text_to_speech(text):
    print("合成语音...")
    communicate = edge_tts.Communicate(
        text=text,
        voice="zh-CN-XiaoxiaoNeural"
    )
    await communicate.save(REPLY_FILE)

# 5. 播放语音（Mac）
def play_audio():
    os.system(f"afplay {REPLY_FILE}")

# 6. 保存为训练数据(JSON)
def save_to_dataset(user_text, ai_text):
    record = {
        "instruction": user_text,
        "input": "",
        "output": ai_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 如果文件存在，追加；否则新建
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("已保存到训练数据集 dataset.json")


# 主流程
def main():
    record_audio(5)
    user_text = speech_to_text()
    if not user_text.strip():
        print("未识别到有效语音")
        return

    reply_text = call_qwen(user_text)
    print("AI 回复:", reply_text)
    # 保存对话为训练数据
    save_to_dataset(user_text, reply_text)

    asyncio.run(text_to_speech(reply_text))
    play_audio()

if __name__ == "__main__":
    main()
