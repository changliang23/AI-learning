from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from models.openai_model import call_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMANEVAL_PATH = PROJECT_ROOT / "datasets" / "humaneval.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "humaneval_report.json"


def estimate_pass_at_k(total: int, correct: int, k: int) -> float:
    if total == 0:
        return 0.0
    if total - correct < k:
        return 1.0
    product = 1.0
    for i in range(total - correct + 1, total + 1):
        product *= 1.0 - k / i
    return 1.0 - product


def load_tasks(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise RuntimeError("HumanEval 数据集必须是 JSON 数组")
    return data


def build_prompt(task: Dict[str, Any]) -> str:
    return (
        "你正在完成 HumanEval 风格的代码生成任务。"
        "请只输出 Python 函数实现代码，不要输出解释，不要加 markdown 代码块。\n\n"
        f"任务ID: {task['task_id']}\n"
        f"题目描述: {task['prompt']}\n\n"
        f"函数签名: {task['entry_point']}\n"
        "请返回完整可执行的 Python 函数实现。"
    )


def evaluate_completion(code: str, tests: List[str]) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {}
    start = time.time()
    try:
        exec(code, namespace, namespace)
        for test in tests:
            exec(test, namespace, namespace)
        return {
            "passed": True,
            "error": "",
            "exec_time_ms": (time.time() - start) * 1000,
        }
    except Exception as error:
        return {
            "passed": False,
            "error": str(error),
            "exec_time_ms": (time.time() - start) * 1000,
        }


def run_humaneval_eval(dataset_path: str, samples_per_task: int = 1) -> Dict[str, Any]:
    tasks = load_tasks(dataset_path)
    results: List[Dict[str, Any]] = []
    total_completions = 0
    total_passed = 0
    total_exec_time = 0.0

    for task in tasks:
        completions: List[Dict[str, Any]] = []
        print(f"[HumanEval] Evaluating {task['task_id']} with {samples_per_task} sample(s)")

        for sample_idx in range(samples_per_task):
            raw_code = call_model(build_prompt(task))
            eval_result = evaluate_completion(raw_code, task["tests"])
            total_completions += 1
            total_exec_time += eval_result["exec_time_ms"]
            if eval_result["passed"]:
                total_passed += 1

            completions.append(
                {
                    "sample_index": sample_idx,
                    "code": raw_code,
                    "passed": eval_result["passed"],
                    "error": eval_result["error"],
                    "exec_time_ms": eval_result["exec_time_ms"],
                }
            )

        correct_for_task = sum(1 for item in completions if item["passed"])
        results.append(
            {
                "task_id": task["task_id"],
                "entry_point": task["entry_point"],
                "prompt": task["prompt"],
                "samples": completions,
                "pass@1": 1.0 if completions and completions[0]["passed"] else 0.0,
                "pass@k": estimate_pass_at_k(samples_per_task, correct_for_task, samples_per_task),
            }
        )

    task_count = len(tasks)
    task_pass_at_1 = sum(item["pass@1"] for item in results) / task_count if task_count else 0.0
    pass_at_k = sum(item["pass@k"] for item in results) / task_count if task_count else 0.0

    return {
        "summary": {
            "dataset_path": str(Path(dataset_path) if Path(dataset_path).is_absolute() else PROJECT_ROOT / dataset_path),
            "task_count": task_count,
            "samples_per_task": samples_per_task,
            "total_completions": total_completions,
            "total_passed": total_passed,
            "pass@1": task_pass_at_1,
            "pass@k": pass_at_k,
            "avg_exec_time_ms": total_exec_time / total_completions if total_completions else 0.0,
        },
        "details": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HumanEval-style evaluation")
    parser.add_argument("--dataset-path", default=str(DEFAULT_HUMANEVAL_PATH))
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    report = run_humaneval_eval(args.dataset_path, args.samples_per_task)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
