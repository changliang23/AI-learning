from __future__ import annotations

import json
from typing import Any, Dict, List

from services.ollama_client import OllamaClient
from agent.tools import TOOL_REGISTRY


class DemoAgent:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.client = OllamaClient(model_name=model_name)

    def _plan(self, task: str, tool_inputs: Dict[str, Any]) -> Dict[str, Any]:
        tool_desc = "\n".join([f"- {name}: {meta['description']}" for name, meta in TOOL_REGISTRY.items()])
        prompt = f"""
你是一个可以调用工具的 Agent 规划器。

任务：{task}
可用工具：
{tool_desc}

默认输入：{json.dumps(tool_inputs, ensure_ascii=False)}

请输出 JSON，格式如下：
{{
  "tool": "工具名",
  "arguments": {{"参数": "值"}},
  "reasoning": "一句话说明为什么调用这个工具",
  "final_instruction": "拿到工具结果后应如何组织最终回答"
}}

不要输出 markdown，只输出 JSON。
""".strip()
        content = self.client.generate(prompt, format="json")
        return json.loads(content)

    def _finalize(self, task: str, plan: Dict[str, Any], tool_result: Dict[str, Any]) -> str:
        prompt = f"""
你是一个执行完成后的 Agent。
任务：{task}
规划：{json.dumps(plan, ensure_ascii=False)}
工具结果：{json.dumps(tool_result, ensure_ascii=False)}

请根据任务和工具结果输出最终中文回答。
""".strip()
        return self.client.generate(prompt)

    def run(self, sample: Dict) -> Dict:
        task = sample["task"]
        inputs = sample["tool_inputs"]
        trace: List[Dict] = []
        plan = self._plan(task, inputs)
        tool_name = plan["tool"]
        tool_args = plan.get("arguments", inputs)
        trace.append({"step": 1, "thought": plan.get("reasoning", "分析任务并选择工具"), "tool": tool_name, "input": tool_args})
        tool_result = TOOL_REGISTRY[tool_name]["handler"](**tool_args)
        trace.append({"step": 2, "observation": tool_result})
        final_answer = self._finalize(task, plan, tool_result)
        trace.append({"step": 3, "thought": plan.get("final_instruction", "根据工具结果组织最终回答")})
        return {
            "model": self.model_name,
            "task": task,
            "trace": trace,
            "final_answer": final_answer,
        }
