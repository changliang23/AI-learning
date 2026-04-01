from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def call_model(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="qwen-turbo",  # 可换 qwen-plus / qwen-max
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()