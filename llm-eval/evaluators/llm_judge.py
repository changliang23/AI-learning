from models.openai_model import call_model

def judge(prompt, answer, expected):
    judge_prompt = f"""
问题: {prompt}
模型回答: {answer}
标准答案: {expected}
请给出0-10分，并说明是否正确。
"""
    return call_model(judge_prompt)