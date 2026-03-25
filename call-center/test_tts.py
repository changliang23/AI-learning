import asyncio
import edge_tts
import os

async def main():
    tts = edge_tts.Communicate("你好，这是测试", voice="zh-CN-XiaoxiaoNeural")
    await tts.save("test.mp3")
    os.system("afplay test.mp3")

asyncio.run(main())
