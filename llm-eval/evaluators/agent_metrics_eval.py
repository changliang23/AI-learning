from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluators.agent_eval import (
    AgentTask,
    AgentResult,
    MockTravelAgent,
    AgentEvaluator,
    compare_agents,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_PATH = PROJECT_ROOT / "datasets" / "agent_tasks.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "agent_report.json"


def load_agent_tasks(dataset_path: str) -> List[AgentTask]:
    path = Path(dataset_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return [AgentTask(**item) for item in data]


def run_agent_eval(
    mode: str = "normal",
    num_runs: int = 1,
    compare: bool = False,
    dataset_path: str = str(DEFAULT_TASKS_PATH),
) -> Dict[str, Any]:
    if compare:
        summary = compare_agents()
        return {
            "summary": summary,
            "mode": "compare",
        }

    tasks = load_agent_tasks(dataset_path)
    agent = MockTravelAgent(mode=mode)
    evaluator = AgentEvaluator()
    all_results: List[AgentResult] = []

    for run_id in range(num_runs):
        print(f"[Agent] Run {run_id + 1}/{num_runs}")
        for task in tasks:
            result = agent.run(task.user_query)
            result.task_id = task.task_id
            all_results.append(result)

    last_results = all_results[-len(tasks):] if all_results else []
    evaluation = evaluator.full_evaluation(last_results, tasks)

    serialized_results = []
    for result in all_results:
        serialized_results.append(
            {
                "task_id": result.task_id,
                "user_query": result.user_query,
                "final_answer": result.final_answer,
                "total_time_ms": result.total_time_ms,
                "total_cost": result.total_cost,
                "tool_calls_summary": result.tool_calls_summary,
                "success": result.success,
                "error_message": result.error_message,
            }
        )

    return {
        "summary": {
            **evaluation,
            "dataset_path": str(Path(dataset_path) if Path(dataset_path).is_absolute() else PROJECT_ROOT / dataset_path),
        },
        "mode": mode,
        "num_runs": num_runs,
        "details": serialized_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument("--mode", choices=["normal", "slow", "error_prone", "inefficient"], default="normal")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--dataset-path", default=str(DEFAULT_TASKS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    report = run_agent_eval(
        mode=args.mode,
        num_runs=args.num_runs,
        compare=args.compare,
        dataset_path=args.dataset_path,
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
