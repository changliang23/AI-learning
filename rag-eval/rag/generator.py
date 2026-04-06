from __future__ import annotations

import json
from typing import Dict, List

from services.ollama_client import OllamaClient


class Generator:
    def __init__(self, model_name: str, judge_model: str | None = None) -> None:
        self.model_name = model_name
        self.client = OllamaClient(model_name=model_name)
        self.judge_client = OllamaClient(model_name=judge_model or model_name)

    def generate(self, question: str, retrieved_contexts: List[str], ground_truth: str) -> Dict:
        prompt = f"""
你是一个 RAG 问答模型。请严格基于提供的上下文回答问题。

问题：{question}

上下文：
{chr(10).join(f'- {item}' for item in retrieved_contexts)}

参考标准答案：{ground_truth}

要求：
1. 用中文回答。
2. 不要编造上下文中不存在的事实。
3. 回答尽量简洁。
""".strip()
        answer = self.client.generate(prompt)
        return {
            "model": self.model_name,
            "question": question,
            "answer": answer,
            "used_context": " ".join(retrieved_contexts),
        }

    def generate_ground_truth(self, question: str, contexts: List[str]) -> str:
        prompt = f"""
你是一个评测基准答案生成器。请只基于给定上下文，为评测问题生成一条尽可能准确、简洁、可核对的标准答案。

问题：{question}

上下文：
{chr(10).join(f'- {item}' for item in contexts)}

要求：
1. 只能使用上下文中的信息。
2. 用中文回答。
3. 长度控制在 1 到 3 句话。
4. 不要解释过程，只输出标准答案正文。
""".strip()
        return self.judge_client.generate(prompt)

    def generate_benchmark_questions(self, contexts: List[str], per_type_count: int = 2) -> List[Dict]:
        prompt = f"""
你是一个 RAG 评测数据集构建器。请基于给定文档内容，自动生成细粒度 benchmark 问题集。

上下文：
{chr(10).join(f'- {item}' for item in contexts)}

要求：
1. 生成四类问题：factual、summary、comparison、reasoning。
2. 每类生成 {per_type_count} 个问题。
3. 返回 JSON 数组，格式如下：
[
  {{"type": "factual", "question": "..."}},
  {{"type": "summary", "question": "..."}}
]
4. 不要输出 markdown，不要附加解释。
""".strip()
        content = self.judge_client.generate(prompt, format="json")
        parsed = json.loads(content)
        results: List[Dict] = []
        for item in parsed:
            if isinstance(item, dict) and item.get("type") and item.get("question"):
                results.append({"type": item["type"], "question": item["question"].strip()})
        return results
