from __future__ import annotations

from typing import Dict


class LLMJudge:
    def judge_rag(self, sample_id: str, metrics: Dict) -> Dict:
        score = metrics["overall"]
        verdict = "strong" if score >= 0.8 else "medium" if score >= 0.6 else "weak"
        return {
            "sample_id": sample_id,
            "score": score,
            "verdict": verdict,
            "comment": f"RAG 回答整体表现为 {verdict}。",
        }

    def judge_agent(self, sample_id: str, metrics: Dict) -> Dict:
        score = metrics["overall"]
        verdict = "strong" if score >= 0.8 else "medium" if score >= 0.6 else "weak"
        return {
            "sample_id": sample_id,
            "score": score,
            "verdict": verdict,
            "comment": f"Agent 执行质量整体表现为 {verdict}。",
        }

