from __future__ import annotations

from typing import Dict, List


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(left.replace("，", " ").replace("。", " ").split())
    right_tokens = set(right.replace("，", " ").replace("。", " ").split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), 1)


class RAGEvaluator:
    def evaluate(self, question: str, contexts: List[str], answer: str, ground_truth: str) -> Dict:
        merged_context = " ".join(contexts)
        context_relevance = round(_token_overlap_score(question, merged_context), 3)
        answer_faithfulness = round(_token_overlap_score(answer, merged_context), 3)
        answer_correctness = round(_token_overlap_score(answer, ground_truth), 3)
        overall = round((context_relevance + answer_faithfulness + answer_correctness) / 3, 3)
        return {
            "Context Relevance": context_relevance,
            "Answer Faithfulness": answer_faithfulness,
            "Answer Correctness": answer_correctness,
            "overall": overall,
        }
