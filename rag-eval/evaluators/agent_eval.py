from __future__ import annotations

from typing import Dict, List


class AgentEvaluator:
    def evaluate(self, trace: List[Dict], final_answer: str, expected_goal: str) -> Dict:
        used_tools = sum(1 for item in trace if item.get("tool"))
        observations = sum(1 for item in trace if item.get("observation"))
        tool_usage = 1.0 if used_tools >= 1 and observations >= 1 else 0.4
        task_completion = 1.0 if any(keyword in final_answer for keyword in ["建议", "约等于", "当前"]) else 0.6
        reasoning = min(1.0, round(len(trace) / 3, 3))
        overall = round((tool_usage + task_completion + reasoning) / 3, 3)
        return {
            "Tool Usage": round(tool_usage, 3),
            "Task Completion": round(task_completion, 3),
            "Reasoning": round(reasoning, 3),
            "overall": overall,
            "expected_goal": expected_goal,
        }

