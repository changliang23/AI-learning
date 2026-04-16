from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluators.agent_metrics_eval import run_agent_eval
from evaluators.benchmark_eval import LOCAL_GSM8K_FILE, LOCAL_MMLU_FILE, run_benchmark_eval
from evaluators.humaneval_eval import DEFAULT_HUMANEVAL_PATH, run_humaneval_eval
from evaluators.llm_judge import judge
from evaluators.rule_eval import exact_match
from evaluators.stability_eval import stability
from models.openai_model import call_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "report.json"
DEFAULT_AGENT_TASKS_PATH = PROJECT_ROOT / "datasets" / "agent_tasks.json"


def run_llm_judge_eval(dataset_path: Path) -> Dict[str, Any]:
    print(f"[LLM Judge] Running task evaluation from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []
    pass_count = 0

    for index, item in enumerate(data, start=1):
        prompt = item["prompt"]
        expected = item["expected"]
        print(f"  - Evaluating sample {index}/{len(data)}: {item['id']}")

        pred = call_model(prompt)
        is_pass = exact_match(pred, expected)
        if is_pass:
            pass_count += 1

        stab_score, outputs = stability(prompt)
        judge_result = json.loads(judge(prompt, pred, expected))

        results.append(
            {
                "id": item["id"],
                "prompt": prompt,
                "pred": pred,
                "expected": expected,
                "pass": is_pass,
                "stability": stab_score,
                "outputs": outputs,
                "judge_score": judge_result["score"],
                "judge_correct": judge_result["correct"],
                "judge_reason": judge_result["reason"],
            }
        )

    metrics = {
        "dataset_path": str(dataset_path),
        "total": len(data),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(data) if data else 0.0,
    }
    return {
        "metrics": metrics,
        "items": results,
    }


def run_rag_eval_block() -> Dict[str, Any]:
    print("[RAG] Running built-in RAG evaluation")
    from evaluators.rag_eval import MockRAGSystem, RAGEvaluator, RAGResult, RAGTestCase

    knowledge_base = {
        "doc_1": "公司年假政策：员工每年享有5天带薪年假。",
        "doc_2": "请假流程：员工请假需提前3天提交申请，经主管批准。",
        "doc_3": "报销流程：差旅费报销需提供发票，填写报销单，财务审核后打款。",
        "doc_4": "加班政策：加班可申请调休或加班费。",
        "doc_5": "社保缴纳：公司为员工缴纳五险一金。"
    }
    test_cases = [
        RAGTestCase("年假有多少天？", "公司年假为每年5天。", ["doc_1"], ["5天", "年假"]),
        RAGTestCase("请假需要提前几天申请？", "请假需提前3天申请。", ["doc_2"], ["提前", "3天"]),
        RAGTestCase("如何报销差旅费？", "差旅费报销需要提供发票，填写报销单，经财务审核后打款。", ["doc_3"], ["发票", "报销单", "财务"]),
        RAGTestCase("公司有哪些福利？", "公司提供年假、五险一金等福利。", ["doc_1", "doc_5"], ["年假", "五险一金"]),
    ]

    rag_system = MockRAGSystem(knowledge_base)
    results: List[RAGResult] = []
    for case in test_cases:
        answer, doc_ids, doc_contents = rag_system.query(case.question)
        results.append(
            RAGResult(
                question=case.question,
                generated_answer=answer,
                retrieved_docs=doc_contents,
                retrieved_doc_ids=doc_ids,
            )
        )

    evaluator = RAGEvaluator()
    metrics = evaluator.full_evaluation(results, test_cases)
    items = [
        {
            "question": result.question,
            "generated_answer": result.generated_answer,
            "retrieved_doc_ids": result.retrieved_doc_ids,
        }
        for result in results
    ]
    return {
        "metrics": metrics,
        "items": items,
    }


def build_report_summary(sections: Dict[str, Any]) -> Dict[str, Any]:
    overview: Dict[str, Any] = {"enabled_sections": list(sections.keys())}

    if "benchmark" in sections:
        metrics = sections["benchmark"].get("metrics", {})
        overview["benchmark"] = {
            "accuracy": metrics.get("accuracy", 0.0),
            "total": metrics.get("total", 0),
            "correct": metrics.get("correct", 0),
            "skipped": metrics.get("skipped", False),
        }

    if "llm_judge" in sections:
        metrics = sections["llm_judge"].get("metrics", {})
        overview["llm_judge"] = {
            "pass_rate": metrics.get("pass_rate", 0.0),
            "pass_count": metrics.get("pass_count", 0),
            "total": metrics.get("total", 0),
        }

    if "humaneval" in sections:
        metrics = sections["humaneval"].get("metrics", {})
        overview["humaneval"] = {
            "pass@1": metrics.get("pass@1", 0.0),
            "pass@k": metrics.get("pass@k", 0.0),
            "task_count": metrics.get("task_count", 0),
            "samples_per_task": metrics.get("samples_per_task", 0),
        }

    if "rag" in sections:
        metrics = sections["rag"].get("metrics", {})
        overview["rag"] = {
            "overall_score": metrics.get("overall_score", 0.0),
            "retrieval_metrics": metrics.get("retrieval_metrics", {}),
            "generation_metrics": metrics.get("generation_metrics", {}),
        }

    if "agent" in sections:
        metrics = sections["agent"].get("metrics", {})
        overview["agent"] = {
            "overall_score": metrics.get("overall_score", 0.0),
            "task_completion_rate": metrics.get("task_completion_rate", 0.0),
            "tool_selection_accuracy": metrics.get("tool_selection_accuracy", 0.0),
            "avg_response_time_ms": metrics.get("avg_response_time_ms", 0.0),
        }

    return overview


def normalize_section(report: Dict[str, Any]) -> Dict[str, Any]:
    if "summary" in report and "details" in report:
        return {
            "metrics": report["summary"],
            "items": report["details"],
        }
    if "summary" in report and "details" not in report:
        data = dict(report)
        metrics = data.pop("summary")
        return {
            "metrics": metrics,
            "items": data,
        }
    return report


def run_eval(
    dataset_path: str,
    benchmark_dataset: str = "all",
    benchmark_limit: int = 10,
    skip_benchmark: bool = False,
    mmlu_dataset_path: Optional[str] = None,
    gsm8k_dataset_path: Optional[str] = None,
    run_benchmark_block: bool = True,
    run_llm_judge_block: bool = True,
    run_humaneval_block: bool = True,
    run_rag_block: bool = True,
    run_agent_block: bool = True,
    humaneval_dataset_path: str = str(DEFAULT_HUMANEVAL_PATH),
    humaneval_samples_per_task: int = 1,
    agent_dataset_path: str = str(DEFAULT_AGENT_TASKS_PATH),
    agent_mode: str = "normal",
    agent_num_runs: int = 1,
    agent_compare: bool = False,
) -> Dict[str, Any]:
    resolved_dataset_path = Path(dataset_path)
    if not resolved_dataset_path.is_absolute():
        resolved_dataset_path = PROJECT_ROOT / resolved_dataset_path

    sections: Dict[str, Any] = {}

    if run_benchmark_block:
        if skip_benchmark:
            sections["benchmark"] = {
                "metrics": {
                    "project_root": str(PROJECT_ROOT),
                    "mode": "local_json",
                    "datasets": [],
                    "total": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                    "skipped": True,
                },
                "items": {},
            }
        else:
            print(f"[Benchmark] Running benchmark evaluation: dataset={benchmark_dataset}, limit={benchmark_limit}")
            sections["benchmark"] = normalize_section(
                run_benchmark_eval(
                    limit=benchmark_limit,
                    dataset=benchmark_dataset,
                    verbose=True,
                    mmlu_dataset_path=mmlu_dataset_path or str(LOCAL_MMLU_FILE),
                    gsm8k_dataset_path=gsm8k_dataset_path or str(LOCAL_GSM8K_FILE),
                )
            )
            sections["benchmark"]["metrics"]["skipped"] = False

    if run_llm_judge_block:
        sections["llm_judge"] = run_llm_judge_eval(resolved_dataset_path)

    if run_humaneval_block:
        print("[HumanEval] Running HumanEval block")
        sections["humaneval"] = normalize_section(
            run_humaneval_eval(humaneval_dataset_path, humaneval_samples_per_task)
        )

    if run_rag_block:
        sections["rag"] = run_rag_eval_block()

    if run_agent_block:
        print("[Agent] Running agent evaluation block")
        sections["agent"] = normalize_section(
            run_agent_eval(
                mode=agent_mode,
                num_runs=agent_num_runs,
                compare=agent_compare,
                dataset_path=agent_dataset_path,
            )
        )

    report = {
        "project_root": str(PROJECT_ROOT),
        "summary": build_report_summary(sections),
        "benchmark": sections.get("benchmark"),
        "llm_judge": sections.get("llm_judge"),
        "humaneval": sections.get("humaneval"),
        "rag": sections.get("rag"),
        "agent": sections.get("agent"),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"Report saved to {REPORT_PATH}")
    return report
