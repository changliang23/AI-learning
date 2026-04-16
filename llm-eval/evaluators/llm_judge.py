from models.openai_model import call_model

def judge(prompt, answer, expected):
    judge_prompt = f"""
    你是一个专业的AI回答裁判，请按照规则评估AI回答的质量。 
评估规则： 
1. 对比AI回答与预期答案，判断AI回答是否正确回答了原始问题 
2. 给出0到10之间的整数：完全正确得10分，部分正确按比例给分，完全错误得0分 
3. 输出严格遵循JSON格式，包含三个字段： 
   - score: 分数，0-10的整数 
   - correct: 是否正确，布尔值（score≥0.8则为true，否则为false） 
   - reason: 判断理由，说明得分依据
问题: {prompt}
模型回答: {answer}
标准答案: {expected}
请给出0-10分，并说明是否正确。
请输出JSON："""
    return call_model(judge_prompt).strip().removeprefix('```json').removesuffix('```').strip()