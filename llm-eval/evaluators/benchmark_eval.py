from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.openai_model import call_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MMLU_FILE = PROJECT_ROOT / "datasets" / "mmlu.json"
LOCAL_GSM8K_FILE = PROJECT_ROOT / "datasets" / "gsm8k.json"


@dataclass
class EvalSampleResult:
    dataset: str
    index: int
    question: str
    prediction: str
    expected: str
    correct: bool
    subject: Optional[str] = None
    raw_response: str = ""


def extract_answer(text: str) -> str:
    content = text.strip()
    patterns = [
        r"最终答案\s*[:：]\s*([A-D])\b",
        r"答案\s*[:：]\s*([A-D])\b",
        r"因此答案是\s*([A-D])\b",
        r"the answer is\s*([A-D])\b",
        r"\b([A-D])\b",
        r"最终答案\s*[:：]\s*(-?\d+(?:\.\d+)?)",
        r"答案\s*[:：]\s*(-?\d+(?:\.\d+)?)",
        r"因此答案是\s*(-?\d+(?:\.\d+)?)",
        r"the answer is\s*(-?\d+(?:\.\d+)?)",
        r"####\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return content


def normalize_number(text: str) -> str:
    value = text.replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return match.group(0) if match else value


def build_mmlu_prompt(sample: Dict[str, Any]) -> str:
    option_lines = [
        f"{label}. {choice}"
        for label, choice in zip(["A", "B", "C", "D"], sample["choices"])
    ]
    options_text = "\n".join(option_lines)
    return (
        "你正在参加 MMLU 多项选择题测试。请阅读题目并只输出一个选项字母（A、B、C 或 D），不要输出解释。\n\n"
        f"题目：{sample['question']}\n"
        f"选项：\n{options_text}\n\n"
        "最终答案："
    )


def build_gsm8k_prompt(sample: Dict[str, Any]) -> str:
    return (
        "你正在参加 GSM8K 数学题测试。请先进行必要的推理，但最后一行必须严格按照“最终答案：数字”的格式输出，"
        "不要输出单位以外的额外总结。\n\n"
        f"题目：{sample['question']}\n"
    )


def _resolve_path(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_local_json_dataset(dataset_path: str, required_keys: List[str]) -> List[Dict[str, Any]]:
    path = _resolve_path(dataset_path)
    if not path.exists():
        raise RuntimeError(f"本地 benchmark 文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise RuntimeError(f"本地 benchmark 文件格式错误，必须是 JSON 数组：{path}")

    for index, item in enumerate(data):
        missing_keys = [key for key in required_keys if key not in item]
        if missing_keys:
            raise RuntimeError(
                f"本地 benchmark 文件缺少字段：{path} 第 {index + 1} 条缺少 {missing_keys}"
            )

    return data


def evaluate_mmlu(limit: int = 10, verbose: bool = False, dataset_path: Optional[str] = None) -> Dict[str, Any]:
    source = dataset_path or str(LOCAL_MMLU_FILE)
    dataset = _load_local_json_dataset(source, ["question", "choices", "answer"])
    answer_map = ["A", "B", "C", "D"]
    results: List[EvalSampleResult] = []
    sample_count = min(limit, len(dataset))

    for index, sample in enumerate(dataset[:sample_count]):
        if verbose:
            print(f"  - [MMLU] sample {index + 1}/{sample_count}: subject={sample.get('subject', 'unknown')}")

        raw_response = call_model(build_mmlu_prompt(sample))
        prediction = extract_answer(raw_response).upper()
        answer = sample["answer"]
        expected = answer if isinstance(answer, str) and answer in answer_map else answer_map[int(answer)]
        correct = prediction == expected

        results.append(
            EvalSampleResult(
                dataset="mmlu",
                index=index,
                question=sample["question"],
                prediction=prediction,
                expected=expected,
                correct=correct,
                subject=sample.get("subject"),
                raw_response=raw_response,
            )
        )

    correct_count = sum(item.correct for item in results)
    accuracy = correct_count / len(results) if results else 0.0
    return {
        "dataset": "mmlu",
        "source": str(_resolve_path(source)),
        "total": len(results),
        "correct": correct_count,
        "accuracy": accuracy,
        "results": [asdict(item) for item in results],
    }


def _extract_gsm8k_expected(answer: str) -> str:
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer)
    if match:
        return match.group(1)
    return normalize_number(answer)


def evaluate_gsm8k(limit: int = 10, verbose: bool = False, dataset_path: Optional[str] = None) -> Dict[str, Any]:
    source = dataset_path or str(LOCAL_GSM8K_FILE)
    dataset = _load_local_json_dataset(source, ["question", "answer"])
    results: List[EvalSampleResult] = []
    sample_count = min(limit, len(dataset))

    for index, sample in enumerate(dataset[:sample_count]):
        if verbose:
            print(f"  - [GSM8K] sample {index + 1}/{sample_count}")

        raw_response = call_model(build_gsm8k_prompt(sample))
        prediction = normalize_number(extract_answer(raw_response))
        expected = normalize_number(_extract_gsm8k_expected(sample["answer"]))
        correct = prediction == expected

        results.append(
            EvalSampleResult(
                dataset="gsm8k",
                index=index,
                question=sample["question"],
                prediction=prediction,
                expected=expected,
                correct=correct,
                raw_response=raw_response,
            )
        )

    correct_count = sum(item.correct for item in results)
    accuracy = correct_count / len(results) if results else 0.0
    return {
        "dataset": "gsm8k",
        "source": str(_resolve_path(source)),
        "total": len(results),
        "correct": correct_count,
        "accuracy": accuracy,
        "results": [asdict(item) for item in results],
    }


def run_benchmark_eval(
    limit: int = 10,
    dataset: str = "all",
    verbose: bool = False,
    mmlu_dataset_path: Optional[str] = None,
    gsm8k_dataset_path: Optional[str] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    if dataset in {"mmlu", "all"}:
        report["mmlu"] = evaluate_mmlu(limit, verbose=verbose, dataset_path=mmlu_dataset_path)

    if dataset in {"gsm8k", "all"}:
        report["gsm8k"] = evaluate_gsm8k(limit, verbose=verbose, dataset_path=gsm8k_dataset_path)

    summary_items = []
    for name in ["mmlu", "gsm8k"]:
        if name in report:
            summary_items.append(
                {
                    "dataset": name,
                    "source": report[name]["source"],
                    "total": report[name]["total"],
                    "correct": report[name]["correct"],
                    "accuracy": report[name]["accuracy"],
                }
            )

    total_samples = sum(item["total"] for item in summary_items)
    total_correct = sum(item["correct"] for item in summary_items)
    overall_accuracy = total_correct / total_samples if total_samples else 0.0

    return {
        "summary": {
            "project_root": str(PROJECT_ROOT),
            "mode": "local_json",
            "datasets": summary_items,
            "total": total_samples,
            "correct": total_correct,
            "accuracy": overall_accuracy,
        },
        "details": report,
    }
