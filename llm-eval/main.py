import argparse
from pathlib import Path

from core.runner import run_eval
from evaluators.agent_metrics_eval import DEFAULT_TASKS_PATH
from evaluators.benchmark_eval import LOCAL_GSM8K_FILE, LOCAL_MMLU_FILE
from evaluators.humaneval_eval import DEFAULT_HUMANEVAL_PATH

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets" / "sample.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation blocks from unified runner")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))

    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--run-llm-judge", action="store_true")
    parser.add_argument("--run-humaneval", action="store_true")
    parser.add_argument("--run-rag", action="store_true")
    parser.add_argument("--run-agent", action="store_true")
    parser.add_argument("--run-all", action="store_true")

    parser.add_argument("--benchmark-dataset", choices=["mmlu", "gsm8k", "all"], default="all")
    parser.add_argument("--benchmark-limit", type=int, default=10)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--mmlu-dataset-path", default=str(LOCAL_MMLU_FILE))
    parser.add_argument("--gsm8k-dataset-path", default=str(LOCAL_GSM8K_FILE))

    parser.add_argument("--humaneval-dataset-path", default=str(DEFAULT_HUMANEVAL_PATH))
    parser.add_argument("--humaneval-samples-per-task", type=int, default=1)

    parser.add_argument("--agent-dataset-path", default=str(DEFAULT_TASKS_PATH))
    parser.add_argument("--agent-mode", choices=["normal", "slow", "error_prone", "inefficient"], default="normal")
    parser.add_argument("--agent-num-runs", type=int, default=1)
    parser.add_argument("--agent-compare", action="store_true")

    args = parser.parse_args()

    run_any = any([
        args.run_benchmark,
        args.run_llm_judge,
        args.run_humaneval,
        args.run_rag,
        args.run_agent,
        args.run_all,
    ])

    run_benchmark_block = args.run_all or args.run_benchmark or not run_any
    run_llm_judge_block = args.run_all or args.run_llm_judge or not run_any
    run_humaneval_block = args.run_all or args.run_humaneval or not run_any
    run_rag_block = args.run_all or args.run_rag or not run_any
    run_agent_block = args.run_all or args.run_agent or not run_any

    report = run_eval(
        dataset_path=args.dataset_path,
        benchmark_dataset=args.benchmark_dataset,
        benchmark_limit=args.benchmark_limit,
        skip_benchmark=args.skip_benchmark,
        mmlu_dataset_path=args.mmlu_dataset_path,
        gsm8k_dataset_path=args.gsm8k_dataset_path,
        run_benchmark_block=run_benchmark_block,
        run_llm_judge_block=run_llm_judge_block,
        run_humaneval_block=run_humaneval_block,
        run_rag_block=run_rag_block,
        run_agent_block=run_agent_block,
        humaneval_dataset_path=args.humaneval_dataset_path,
        humaneval_samples_per_task=args.humaneval_samples_per_task,
        agent_dataset_path=args.agent_dataset_path,
        agent_mode=args.agent_mode,
        agent_num_runs=args.agent_num_runs,
        agent_compare=args.agent_compare,
    )
    print("Evaluation Done")
    print(report)
