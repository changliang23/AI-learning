import whisper
import torch
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import edge_tts
import asyncio

# ===== 模型路径 =====
BASE_MODEL = "./qwen2.5-1.5b"
LORA_MODEL = "./lora-model"

# ===== 加载模型 =====
asr_model = whisper.load_model("base")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = PeftModel.from_pretrained(model, LORA_MODEL)
model.eval()

# ===== 录音 =====
def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎙 开始说话...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("✅ 录音完成")

# ===== 语音转文字 =====
def speech_to_text():
    result = asr_model.transcribe("input.wav")
    print("📝 识别结果:", result["text"])
    return result["text"]

# ===== 大模型推理 =====
def llm_reply(text):
    prompt = f"用户：{text}\n助手："
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    print("🤖 模型回复:", reply)
    return reply

# ===== 文本转语音 =====
async def text_to_speech(text):
    communicate = edge_tts.Communicate(text=text, voice="zh-CN-XiaoxiaoNeural")
    await communicate.save("reply.mp3")
    print("🔊 已生成语音 reply.mp3")

# ===== 主流程 =====
def main():
    record_audio()
    text = speech_to_text()
    reply = llm_reply(text)
    asyncio.run(text_to_speech(reply))

if __name__ == "__main__":
    main()
