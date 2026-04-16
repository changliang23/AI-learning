Quick Start:
1. pip install -r requirements.txt
2. 确保本地模型服务可用（当前 `models/openai_model.py` 默认连到 `http://localhost:11434/v1`）
3. 运行统一入口评测:
   python main.py --run-all

项目根目录:
- 本项目统一以 `../AI-learning/llm-eval` 为根节点
- 默认任务数据集路径: `datasets/sample.json`
- 默认本地 benchmark 文件:
  - MMLU: `datasets/mmlu.json`
  - GSM8K: `datasets/gsm8k.json`
  - HumanEval: `datasets/humaneval.json`
  - Agent: `datasets/agent_tasks.json`
- 默认报告输出路径: `reports/report.json`

统一 runner 入口:
- 所有测评块都已接入 `core/runner.py`
- 报告结构已整理为更扁平的顶层分区，便于直接查看:
  - `summary`
  - `benchmark`
  - `llm_judge`
  - `humaneval`
  - `rag`
  - `agent`

可选运行配置:
- 全量运行:
  python main.py --run-all
- 只跑 benchmark:
  python main.py --run-benchmark
- 只跑 llm_judge:
  python main.py --run-llm-judge
- 只跑 humaneval:
  python main.py --run-humaneval
- 只跑 rag:
  python main.py --run-rag
- 只跑 agent:
  python main.py --run-agent
- 如果不加任何 `--run-*` 参数，默认全部运行

Benchmark 本地模式:
- `datasets/mmlu.json`
- `datasets/gsm8k.json`
- 常用参数:
  - `--benchmark-dataset all|mmlu|gsm8k`
  - `--benchmark-limit 20`
  - `--skip-benchmark`
  - `--mmlu-dataset-path datasets/mmlu.json`
  - `--gsm8k-dataset-path datasets/gsm8k.json`

LLM Judge:
- 使用 `datasets/sample.json`
- 常用参数:
  - `--dataset-path datasets/sample.json`

HumanEval:
- 使用 `datasets/humaneval.json`
- 常用参数:
  - `--humaneval-dataset-path datasets/humaneval.json`
  - `--humaneval-samples-per-task 3`
- 主要指标:
  - `pass@1`
  - `pass@k`
  - `total_passed`
  - `avg_exec_time_ms`

RAG:
- 当前使用内置 mock knowledge base 和测试集
- 主要指标:
  - `hit_rate`
  - `mrr`
  - `precision@1`
  - `precision@3`
  - `recall@5`
  - `exact_match`
  - `keyword_coverage`
  - `answer_relevance`
  - `bleu`
  - `overall_score`

Agent:
- 使用 `datasets/agent_tasks.json`
- 常用参数:
  - `--agent-dataset-path datasets/agent_tasks.json`
  - `--agent-mode normal|slow|error_prone|inefficient`
  - `--agent-num-runs 3`
  - `--agent-compare`
- 主要指标:
  - `task_completion_rate`
  - `tool_selection_accuracy`
  - `tool_sequence_accuracy`
  - `exact_match_rate`
  - `avg_keyword_coverage`
  - `avg_response_time_ms`
  - `avg_cost_per_task`
  - `cost_per_success`
  - `pass@1_avg`
  - `pass@3_avg`
  - `pass^2_avg`
  - `trajectory_metrics`
  - `overall_score`

报告输出:
- 统一写入 `reports/report.json`
- 顶层新增 `summary` 总览区，便于快速查看关键指标
- 每个测评块直接放在顶层，减少目录深度
- 结构示意:
  - `project_root`
  - `summary`
    - `enabled_sections`
    - `benchmark.accuracy`
    - `llm_judge.pass_rate`
    - `humaneval.pass@1`
    - `rag.overall_score`
    - `agent.overall_score`
  - `benchmark`
    - `metrics`
    - `items`
  - `llm_judge`
    - `metrics`
    - `items`
  - `humaneval`
    - `metrics`
    - `items`
  - `rag`
    - `metrics`
    - `items`
  - `agent`
    - `metrics`
    - `items`

运行时日志:
- 每个评测块会单独打印自己的执行进度
