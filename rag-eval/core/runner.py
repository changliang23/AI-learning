from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import mean
from typing import Dict, List

from agent.agent import DemoAgent
from evaluators.agent_eval import AgentEvaluator
from evaluators.llm_judge import LLMJudge
from evaluators.rag_eval import RAGEvaluator
from rag.generator import Generator
from rag.retriever import QdrantRetriever

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports"
HISTORY_DIR = REPORTS_DIR / "history"


class EvaluationRunner:
    def __init__(self, model_names: List[str] | None = None, judge_model: str | None = None) -> None:
        self.model_names = model_names or ["qwen3:8b"]
        self.judge_model = judge_model or self.model_names[0]
        self.rag_evaluator = RAGEvaluator()
        self.agent_evaluator = AgentEvaluator()
        self.judge = LLMJudge()

    def _load_json(self, path: Path) -> List[Dict]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _save_json(self, path: Path, payload: Dict | List) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def build_test_datasets(self) -> Dict[str, List[Dict]]:
        rag_samples = self._load_json(DATASETS_DIR / "rag.json")
        agent_samples = self._load_json(DATASETS_DIR / "agent.json")
        return {"rag": rag_samples, "agent": agent_samples}

    def generate_benchmark_from_documents(self, documents: List[str], per_type_count: int = 2) -> List[Dict]:
        generator = Generator(self.model_names[0], judge_model=self.judge_model)
        combined_context = documents[: min(len(documents), 8)]
        return generator.generate_benchmark_questions(combined_context, per_type_count=per_type_count)

    def build_uploaded_rag_dataset(self, documents: List[str], evaluation_questions: List[Dict]) -> List[Dict]:
        generator = Generator(self.model_names[0], judge_model=self.judge_model)
        samples: List[Dict] = []
        for index, item in enumerate(evaluation_questions, start=1):
            question = item["question"]
            ground_truth = generator.generate_ground_truth(question, documents)
            samples.append(
                {
                    "id": f"upload-rag-{index}",
                    "question": question,
                    "question_type": item.get("type", "general"),
                    "ground_truth": ground_truth,
                    "contexts": documents,
                }
            )
        return samples

    def summarize_benchmark(self, rag_results: List[Dict]) -> Dict:
        benchmark_summary: Dict[str, Dict[str, float | int]] = {}
        grouped: Dict[str, List[float]] = {}
        for item in rag_results:
            question_type = item.get("question_type", "general")
            grouped.setdefault(question_type, []).append(item["metrics"]["overall"])
        for question_type, scores in grouped.items():
            benchmark_summary[question_type] = {
                "count": len(scores),
                "avg_score": round(sum(scores) / len(scores), 3),
            }
        return benchmark_summary

    def run_rag(self, rag_samples: List[Dict]) -> List[Dict]:
        all_docs = [ctx for item in rag_samples for ctx in item["contexts"]]
        retriever = QdrantRetriever()
        retriever.rebuild_collection(all_docs)
        results: List[Dict] = []
        for model_name in self.model_names:
            generator = Generator(model_name, judge_model=self.judge_model)
            for sample in rag_samples:
                retrieved = retriever.search(sample["question"], top_k=3)
                retrieved_contexts = [item.text for item in retrieved]
                generation = generator.generate(sample["question"], retrieved_contexts, sample["ground_truth"])
                metrics = self.rag_evaluator.evaluate(
                    sample["question"],
                    retrieved_contexts,
                    generation["answer"],
                    sample["ground_truth"],
                )
                judgment = self.judge.judge_rag(sample["id"], metrics)
                results.append(
                    {
                        "sample_id": sample["id"],
                        "model": model_name,
                        "question": sample["question"],
                        "question_type": sample.get("question_type", "general"),
                        "ground_truth": sample["ground_truth"],
                        "retrieved_contexts": retrieved_contexts,
                        "answer": generation["answer"],
                        "metrics": metrics,
                        "judge": judgment,
                    }
                )
        return results

    def run_agent(self, agent_samples: List[Dict]) -> List[Dict]:
        results: List[Dict] = []
        for model_name in self.model_names:
            agent = DemoAgent(model_name)
            for sample in agent_samples:
                execution = agent.run(sample)
                metrics = self.agent_evaluator.evaluate(
                    execution["trace"], execution["final_answer"], sample["expected_goal"]
                )
                judgment = self.judge.judge_agent(sample["id"], metrics)
                results.append(
                    {
                        "sample_id": sample["id"],
                        "model": model_name,
                        "task": sample["task"],
                        "trace": execution["trace"],
                        "final_answer": execution["final_answer"],
                        "metrics": metrics,
                        "judge": judgment,
                    }
                )
        return results

    def save_history(self, report: Dict, task_type: str, metadata: Dict | None = None) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        history_path = HISTORY_DIR / f"{task_type}-{timestamp}.json"
        payload = {
            "task_type": task_type,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "metadata": metadata or {},
            "report": report,
        }
        self._save_json(history_path, payload)
        return history_path

    def list_history(self) -> List[Dict]:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        items: List[Dict] = []
        for path in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            items.append(
                {
                    "file_name": path.name,
                    "path": str(path),
                    "task_type": payload.get("task_type", "unknown"),
                    "created_at": payload.get("created_at", ""),
                    "models": payload.get("report", {}).get("summary", {}).get("models_compared", []),
                    "judge_model": payload.get("report", {}).get("summary", {}).get("judge_model", ""),
                }
            )
        return items

    def load_history(self, file_name: str) -> Dict:
        path = HISTORY_DIR / file_name
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def delete_history(self, file_name: str) -> None:
        path = HISTORY_DIR / file_name
        if path.exists():
            path.unlink()

    def delete_histories(self, file_names: List[str]) -> int:
        deleted = 0
        for file_name in file_names:
            path = HISTORY_DIR / file_name
            if path.exists():
                path.unlink()
                deleted += 1
        return deleted

    def clear_history(self) -> int:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        files = list(HISTORY_DIR.glob("*.json"))
        for path in files:
            path.unlink()
        return len(files)

    def export_report_markdown(self, payload: Dict) -> str:
        report = payload.get("report", payload)
        summary = report.get("summary", {})
        lines = [
            "# Evaluation Report",
            "",
            f"- Models: {', '.join(summary.get('models_compared', []))}",
            f"- Judge Model: {summary.get('judge_model', '')}",
            f"- RAG Avg: {summary.get('rag_avg', 0)}",
            f"- Agent Avg: {summary.get('agent_avg', 0)}",
            "",
            "## Benchmark Summary",
        ]
        for key, value in report.get("benchmark_summary", {}).items():
            lines.append(f"- {key}: count={value.get('count', 0)}, avg_score={value.get('avg_score', 0)}")
        return "\n".join(lines)

    def export_report_html(self, payload: Dict) -> str:
        report = payload.get("report", payload)
        summary = report.get("summary", {})
        benchmark_items = report.get("benchmark_summary", {})
        rag_results = report.get("rag_results", [])
        agent_results = report.get("agent_results", [])
        model_names = summary.get("models_compared", [])

        def safe_text(value: object) -> str:
            return escape(str(value or ""))

        def to_score(value: object) -> float:
            try:
                return round(float(value), 3)
            except (TypeError, ValueError):
                return 0.0

        def average(values: List[float]) -> float:
            return round(sum(values) / len(values), 3) if values else 0.0

        def polar_to_xy(cx: float, cy: float, radius: float, angle_deg: float) -> tuple[float, float]:
            from math import cos, radians, sin

            angle = radians(angle_deg - 90)
            return cx + radius * cos(angle), cy + radius * sin(angle)

        rag_metric_names = ["Context Relevance", "Answer Faithfulness", "Answer Correctness"]
        agent_metric_names = ["Tool Usage", "Task Completion", "Reasoning"]
        palette = ["#3ae374", "#35d0ff", "#ffb627", "#ff5d8f", "#9d7bff", "#7cf5d6"]

        model_rag_summary: List[Dict[str, object]] = []
        model_agent_summary: List[Dict[str, object]] = []
        for index, model_name in enumerate(model_names):
            model_rag_items = [item for item in rag_results if item.get("model") == model_name]
            model_agent_items = [item for item in agent_results if item.get("model") == model_name]
            rag_metrics = {
                metric: average([to_score(item.get("metrics", {}).get(metric, 0)) for item in model_rag_items])
                for metric in rag_metric_names
            }
            agent_metrics = {
                metric: average([to_score(item.get("metrics", {}).get(metric, 0)) for item in model_agent_items])
                for metric in agent_metric_names
            }
            model_rag_summary.append(
                {
                    "model": model_name,
                    "color": palette[index % len(palette)],
                    "overall": average([to_score(item.get("metrics", {}).get("overall", 0)) for item in model_rag_items]),
                    "metrics": rag_metrics,
                }
            )
            model_agent_summary.append(
                {
                    "model": model_name,
                    "color": palette[index % len(palette)],
                    "overall": average([to_score(item.get("metrics", {}).get("overall", 0)) for item in model_agent_items]),
                    "metrics": agent_metrics,
                }
            )

        radar_size = 360
        radar_center = 180
        radar_radius = 120
        radar_axes = len(rag_metric_names + agent_metric_names)
        radar_grid = []
        for ring_ratio in [0.25, 0.5, 0.75, 1.0]:
            points = []
            for axis_index in range(radar_axes):
                angle = axis_index * (360 / radar_axes)
                x, y = polar_to_xy(radar_center, radar_center, radar_radius * ring_ratio, angle)
                points.append(f"{x:.1f},{y:.1f}")
            radar_grid.append(f"<polygon points='{' '.join(points)}' fill='none' stroke='rgba(255,255,255,0.09)' stroke-width='1' />")
        radar_axis_lines = []
        radar_axis_labels = []
        radar_axis_names = rag_metric_names + agent_metric_names
        for axis_index, axis_name in enumerate(radar_axis_names):
            angle = axis_index * (360 / radar_axes)
            x, y = polar_to_xy(radar_center, radar_center, radar_radius, angle)
            label_x, label_y = polar_to_xy(radar_center, radar_center, radar_radius + 28, angle)
            radar_axis_lines.append(
                f"<line x1='{radar_center}' y1='{radar_center}' x2='{x:.1f}' y2='{y:.1f}' stroke='rgba(255,255,255,0.12)' stroke-width='1' />"
            )
            radar_axis_labels.append(
                f"<text x='{label_x:.1f}' y='{label_y:.1f}' text-anchor='middle' fill='#b4d3ff' font-size='12'>{safe_text(axis_name)}</text>"
            )
        radar_polygons = []
        radar_legend = []
        for item in model_rag_summary:
            merged_scores = [item["metrics"].get(metric, 0) for metric in rag_metric_names]
            agent_match = next((entry for entry in model_agent_summary if entry["model"] == item["model"]), None)
            merged_scores.extend(agent_match["metrics"].get(metric, 0) if agent_match else 0 for metric in agent_metric_names)
            points = []
            for axis_index, score in enumerate(merged_scores):
                angle = axis_index * (360 / radar_axes)
                x, y = polar_to_xy(radar_center, radar_center, radar_radius * (to_score(score) / 5), angle)
                points.append(f"{x:.1f},{y:.1f}")
            color = item["color"]
            radar_polygons.append(
                f"<polygon points='{' '.join(points)}' fill='{color}22' stroke='{color}' stroke-width='2.5' />"
            )
            radar_legend.append(
                f"<div class='legend-item'><span class='dot' style='background:{color}'></span><span>{safe_text(item['model'])} · 总分 {item['overall']}</span></div>"
            )

        rag_chart_width = 720
        rag_bar_rows = []
        rag_axis_labels = []
        rag_max_width = 480
        for index, metric in enumerate(rag_metric_names):
            y = 34 + index * 64
            rag_axis_labels.append(f"<text x='14' y='{y + 18}' fill='#b4d3ff' font-size='13'>{safe_text(metric)}</text>")
            for model_index, item in enumerate(model_rag_summary):
                offset = model_index * 18
                value = to_score(item["metrics"].get(metric, 0))
                width = (value / 5) * rag_max_width
                rag_bar_rows.append(
                    f"<rect x='180' y='{y + offset}' width='{width:.1f}' height='14' rx='7' fill='{item['color']}' opacity='0.92' />"
                    f"<text x='{190 + width:.1f}' y='{y + offset + 11}' fill='#ebf4ff' font-size='12'>{value}</text>"
                )

        agent_chart_width = 720
        agent_chart_height = 270
        agent_baseline_y = 220
        agent_step_x = 200
        agent_path_lines = []
        agent_points = []
        agent_labels = []
        for metric_index, metric in enumerate(agent_metric_names):
            x = 120 + metric_index * agent_step_x
            agent_labels.append(f"<text x='{x}' y='244' text-anchor='middle' fill='#b4d3ff' font-size='13'>{safe_text(metric)}</text>")
            agent_labels.append(f"<line x1='{x}' y1='48' x2='{x}' y2='{agent_baseline_y}' stroke='rgba(255,255,255,0.08)' stroke-width='1' />")
        for item in model_agent_summary:
            coords = []
            for metric_index, metric in enumerate(agent_metric_names):
                x = 120 + metric_index * agent_step_x
                value = to_score(item["metrics"].get(metric, 0))
                y = agent_baseline_y - (value / 5) * 150
                coords.append((x, y, value))
            if coords:
                path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
                agent_path_lines.append(
                    f"<polyline points='{path}' fill='none' stroke='{item['color']}' stroke-width='3' stroke-linecap='round' stroke-linejoin='round' />"
                )
                for x, y, value in coords:
                    agent_points.append(
                        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5.5' fill='{item['color']}' />"
                        f"<text x='{x:.1f}' y='{y - 12:.1f}' text-anchor='middle' fill='#ebf4ff' font-size='12'>{value}</text>"
                    )

        chart_max = max([value.get("avg_score", 0) for value in benchmark_items.values()] or [1])
        bar_gap = 22
        bar_width = 120
        base_y = 210
        benchmark_bars = []
        benchmark_labels = []
        for index, (key, value) in enumerate(benchmark_items.items()):
            bar_height = 0 if chart_max == 0 else (value.get("avg_score", 0) / chart_max) * 150
            x = 48 + index * (bar_width + bar_gap)
            y = base_y - bar_height
            benchmark_bars.append(
                f"<rect x='{x}' y='{y:.1f}' width='{bar_width}' height='{bar_height:.1f}' rx='14' fill='url(#barGradient)' />"
                f"<text x='{x + bar_width / 2}' y='{y - 10:.1f}' text-anchor='middle' fill='#e9f2ff' font-size='14'>{value.get('avg_score', 0)}</text>"
            )
            benchmark_labels.append(
                f"<text x='{x + bar_width / 2}' y='{base_y + 24}' text-anchor='middle' fill='#9fc3ff' font-size='13'>{safe_text(key)}</text>"
            )

        donut_segments = []
        donut_legend_items = []
        total_count = sum(value.get("count", 0) for value in benchmark_items.values()) or 1
        current_offset = 0.0
        circumference = 2 * 3.14159 * 70
        for index, (key, value) in enumerate(benchmark_items.items()):
            portion = value.get("count", 0) / total_count
            segment = circumference * portion
            color = palette[index % len(palette)]
            donut_segments.append(
                f"<circle cx='110' cy='110' r='70' fill='none' stroke='{color}' stroke-width='20' stroke-dasharray='{segment:.1f} {circumference - segment:.1f}' stroke-dashoffset='{-current_offset:.1f}' transform='rotate(-90 110 110)' stroke-linecap='round' />"
            )
            current_offset += segment
            donut_legend_items.append(
                f"<div class='legend-item'><span class='dot' style='background:{color}'></span><span>{safe_text(key)} · {value.get('count', 0)}</span></div>"
            )

        rag_rows = "".join(
            f"<tr><td>{safe_text(item.get('model', ''))}</td><td>{safe_text(item.get('sample_id', ''))}</td><td>{safe_text(item.get('question_type', 'general'))}</td><td>{safe_text(item.get('question', ''))}</td><td>{to_score(item.get('metrics', {}).get('overall', 0))}</td></tr>"
            for item in rag_results[:30]
        )
        benchmark_cards = "".join(
            f"<div class='card'><h3>{safe_text(key)}</h3><p>count: {value.get('count', 0)}</p><p>avg: {value.get('avg_score', 0)}</p></div>"
            for key, value in benchmark_items.items()
        )
        embedded_payload = safe_text(json.dumps(payload, ensure_ascii=False, indent=2))

        return f"""
<!DOCTYPE html>
<html lang='zh-CN'>
<head>
  <meta charset='UTF-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1.0' />
  <title>Evaluation Report</title>
  <style>
    :root {{
      --bg: #04111f;
      --panel: rgba(9, 18, 36, 0.92);
      --panel-soft: rgba(15, 29, 51, 0.92);
      --text: #ecf5ff;
      --muted: #9cb9de;
      --line: rgba(255,255,255,0.08);
      --accent: #3ae374;
      --accent-2: #35d0ff;
      --accent-3: #ffb627;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      color: var(--text);
      font-family: 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', sans-serif;
      background:
        radial-gradient(circle at top left, rgba(58, 227, 116, 0.14), transparent 22%),
        radial-gradient(circle at top right, rgba(53, 208, 255, 0.18), transparent 24%),
        linear-gradient(135deg, #04111f, #08182b 45%, #030812);
    }}
    .shell {{ max-width: 1480px; margin: 0 auto; }}
    .hero {{
      background: linear-gradient(180deg, rgba(12,24,43,0.95), rgba(7,14,27,0.96));
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 26px;
      box-shadow: 0 30px 100px rgba(0,0,0,0.35);
    }}
    .hero h1 {{ margin: 0 0 10px; font-size: 34px; letter-spacing: 0.02em; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-top: 20px; }}
    .section-grid {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 18px; margin-top: 24px; }}
    .double-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin-top: 24px; }}
    .card, .panel {{
      background: linear-gradient(180deg, var(--panel-soft), var(--panel));
      border: 1px solid rgba(58, 227, 116, 0.14);
      border-radius: 20px;
      padding: 18px;
      overflow: hidden;
    }}
    h2, h3 {{ margin: 0 0 10px; }}
    .legend-item {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; color: #d4e8ff; font-size: 14px; }}
    .dot {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; box-shadow: 0 0 16px currentColor; }}
    .chart-caption {{ color: var(--muted); font-size: 13px; margin-top: 10px; line-height: 1.6; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 24px; background: rgba(255,255,255,0.025); border-radius: 16px; overflow: hidden; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ color: var(--accent); font-weight: 600; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; color: #dcecff; font-size: 12px; line-height: 1.55; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }}
    .badge {{ padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.06); border: 1px solid var(--line); color: #cde3ff; font-size: 13px; }}
    .footer-note {{ margin-top: 22px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 1100px) {{
      body {{ padding: 18px; }}
      .section-grid, .double-grid {{ grid-template-columns: 1fr; }}
      svg {{ width: 100%; height: auto; }}
    }}
  </style>
</head>
<body>
  <div class='shell'>
    <section class='hero'>
      <h1>RAG & Agent Evaluation Report</h1>
      <p class='muted'>单文件自包含 HTML 报告，无需外部 JS / CSS 依赖，可直接发送、归档或离线打开。</p>
      <div class='badge-row'>
        <span class='badge'>Models: {safe_text(', '.join(model_names))}</span>
        <span class='badge'>Judge: {safe_text(summary.get('judge_model', ''))}</span>
        <span class='badge'>RAG Avg: {summary.get('rag_avg', 0)}</span>
        <span class='badge'>Agent Avg: {summary.get('agent_avg', 0)}</span>
        <span class='badge'>RAG Samples: {len(rag_results)}</span>
        <span class='badge'>Agent Samples: {len(agent_results)}</span>
      </div>
      <div class='grid'>
        <div class='card'><h3>Judge Model</h3><p>{safe_text(summary.get('judge_model', ''))}</p></div>
        <div class='card'><h3>RAG 平均分</h3><p>{summary.get('rag_avg', 0)}</p></div>
        <div class='card'><h3>Agent 平均分</h3><p>{summary.get('agent_avg', 0)}</p></div>
        <div class='card'><h3>Benchmark 类型数</h3><p>{len(benchmark_items)}</p></div>
      </div>
    </section>

    <div class='section-grid'>
      <section class='panel'>
        <h2>模型对比雷达图</h2>
        <svg width='{radar_size}' height='{radar_size}' viewBox='0 0 {radar_size} {radar_size}' role='img' aria-label='模型对比雷达图'>
          {''.join(radar_grid)}
          {''.join(radar_axis_lines)}
          {''.join(radar_polygons)}
          {''.join(radar_axis_labels)}
          <circle cx='{radar_center}' cy='{radar_center}' r='3' fill='#ecf5ff' />
        </svg>
        <div>{''.join(radar_legend) or "<p class='muted'>暂无模型指标数据。</p>"}</div>
        <div class='chart-caption'>将 RAG 三维指标与 Agent 三维指标合并到一张雷达图中，便于快速对比不同模型的综合轮廓。</div>
      </section>

      <section class='panel'>
        <h2>Question Type Mix</h2>
        <svg width='220' height='220' viewBox='0 0 220 220' role='img' aria-label='问题类型占比图'>
          <circle cx='110' cy='110' r='70' fill='none' stroke='rgba(255,255,255,0.08)' stroke-width='20' />
          {''.join(donut_segments)}
          <text x='110' y='104' text-anchor='middle' fill='#e9f2ff' font-size='26' font-weight='700'>{total_count}</text>
          <text x='110' y='128' text-anchor='middle' fill='#9fc3ff' font-size='13'>questions</text>
        </svg>
        <div>{''.join(donut_legend_items) or "<p class='muted'>No type data.</p>"}</div>
        <div class='chart-caption'>环形图展示细粒度 benchmark 的题型分布，用于观察问题集结构是否均衡。</div>
      </section>
    </div>

    <div class='double-grid'>
      <section class='panel'>
        <h2>RAG 三维指标横向条形图</h2>
        <svg width='{rag_chart_width}' height='240' viewBox='0 0 {rag_chart_width} 240' role='img' aria-label='RAG 三维指标横向条形图'>
          <line x1='180' y1='18' x2='180' y2='216' stroke='rgba(255,255,255,0.16)' stroke-width='1' />
          <line x1='660' y1='18' x2='660' y2='216' stroke='rgba(255,255,255,0.06)' stroke-width='1' />
          {''.join(rag_axis_labels)}
          {''.join(rag_bar_rows)}
        </svg>
        <div>{''.join(radar_legend) or "<p class='muted'>暂无 RAG 数据。</p>"}</div>
        <div class='chart-caption'>按指标横向比较不同模型在 Context Relevance、Answer Faithfulness、Answer Correctness 三项上的平均分。</div>
      </section>

      <section class='panel'>
        <h2>Agent 执行评分图</h2>
        <svg width='{agent_chart_width}' height='{agent_chart_height}' viewBox='0 0 {agent_chart_width} {agent_chart_height}' role='img' aria-label='Agent 执行评分图'>
          <line x1='72' y1='{agent_baseline_y}' x2='{agent_chart_width - 50}' y2='{agent_baseline_y}' stroke='rgba(255,255,255,0.16)' stroke-width='1' />
          {''.join(agent_labels)}
          {''.join(agent_path_lines)}
          {''.join(agent_points)}
        </svg>
        <div>{''.join(radar_legend) or "<p class='muted'>暂无 Agent 数据。</p>"}</div>
        <div class='chart-caption'>折线图按 Tool Usage、Task Completion、Reasoning 三个维度展示各模型 Agent 执行得分趋势。</div>
      </section>
    </div>

    <h2 style='margin-top: 28px;'>Benchmark Summary</h2>
    <div class='grid'>{benchmark_cards or "<div class='card'><p>No benchmark summary.</p></div>"}</div>

    <div class='section-grid'>
      <section class='panel'>
        <h2>Benchmark Avg Score Chart</h2>
        <svg width='680' height='240' viewBox='0 0 680 240' role='img' aria-label='Benchmark 平均分柱状图'>
          <defs>
            <linearGradient id='barGradient' x1='0' x2='0' y1='0' y2='1'>
              <stop offset='0%' stop-color='#35d0ff' />
              <stop offset='100%' stop-color='#3ae374' />
            </linearGradient>
          </defs>
          <line x1='36' y1='210' x2='656' y2='210' stroke='rgba(255,255,255,0.2)' stroke-width='1' />
          {''.join(benchmark_bars)}
          {''.join(benchmark_labels)}
        </svg>
        <div class='chart-caption'>柱状图展示不同 benchmark 类型的平均得分，可用于观察题型难度差异。</div>
      </section>

      <section class='panel'>
        <h2>报告原始数据</h2>
        <pre>{embedded_payload}</pre>
        <div class='chart-caption'>导出的 HTML 已内嵌所有样式与数据，打开文件即可查看，不依赖外部资源。</div>
      </section>
    </div>

    <h2 style='margin-top:28px;'>Top RAG Results</h2>
    <table>
      <thead><tr><th>Model</th><th>Sample</th><th>Type</th><th>Question</th><th>Overall</th></tr></thead>
      <tbody>{rag_rows or "<tr><td colspan='5'>No data</td></tr>"}</tbody>
    </table>

    <div class='footer-note'>报告生成时间：{safe_text(datetime.now().isoformat(timespec='seconds'))}</div>
  </div>
</body>
</html>
""".strip()

    def compile_report(self, save_history: bool = True) -> Dict:
        datasets = self.build_test_datasets()
        rag_results = self.run_rag(datasets["rag"])
        agent_results = self.run_agent(datasets["agent"])
        report = {
            "summary": {
                "models_compared": self.model_names,
                "judge_model": self.judge_model,
                "rag_avg": round(mean(item["metrics"]["overall"] for item in rag_results), 3),
                "agent_avg": round(mean(item["metrics"]["overall"] for item in agent_results), 3),
            },
            "benchmark_summary": self.summarize_benchmark(rag_results),
            "rag_results": rag_results,
            "agent_results": agent_results,
        }
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._save_json(REPORTS_DIR / "report.json", report)
        if save_history:
            self.save_history(report, task_type="platform-overview")
        return report
