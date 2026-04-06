from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
from docx import Document
from pypdf import PdfReader

from core.runner import EvaluationRunner
from services.ollama_client import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = [
    model.strip()
    for model in os.getenv("AVAILABLE_OLLAMA_MODELS", "qwen3:8b,llama3.1:8b,deepseek-r1:8b").split(",")
    if model.strip()
]
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_MODELS[0] if DEFAULT_MODELS else "qwen3:8b")

st.set_page_config(page_title="RAG & Agent Evaluation Platform", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(0, 255, 163, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 153, 255, 0.12), transparent 24%),
            linear-gradient(135deg, #07111f 0%, #0c1728 45%, #050914 100%);
        color: #e9f2ff;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        background: rgba(7, 17, 31, 0.75);
        box-shadow: 0 24px 80px rgba(0,0,0,0.28);
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(16,28,46,0.95), rgba(10,18,32,0.95));
        border: 1px solid rgba(0,255,163,0.14);
    }
    .trace-box {
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid #00ffa3;
        border-radius: 12px;
        background: rgba(255,255,255,0.04);
        font-family: 'IBM Plex Mono', monospace;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_text_from_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_file.read().decode("utf-8")
    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        document = Document(uploaded_file)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise ValueError("仅支持 txt / pdf / docx 文件")


def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    cleaned = " ".join(text.split())
    return [cleaned[i:i + chunk_size] for i in range(0, len(cleaned), chunk_size) if cleaned[i:i + chunk_size]]


@st.cache_data(show_spinner=False)
def run_demo(model_names: List[str], judge_model: str) -> dict:
    runner = EvaluationRunner(model_names=model_names, judge_model=judge_model)
    return runner.compile_report(save_history=True)


def render_dashboard(report: dict) -> None:
    rag_df = pd.DataFrame([
        {
            "sample_id": item["sample_id"],
            "model": item["model"],
            "question_type": item.get("question_type", "general"),
            **item["metrics"],
        }
        for item in report["rag_results"]
    ])
    agent_df = pd.DataFrame([
        {
            "sample_id": item["sample_id"],
            "model": item["model"],
            **item["metrics"],
        }
        for item in report["agent_results"]
    ])

    st.markdown(
        """
        <div class="hero">
            <h1>RAG & Agent Evaluation Platform</h1>
            <p>接入本地 Ollama 与 Qdrant 的 AI 应用测评平台，支持历史记录管理、结果导出、细粒度 benchmark 自动生成、多模型横向对比、RAG 打分与 Agent 执行轨迹展示。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><h3>模型对比</h3><h2>{len(report['summary']['models_compared'])}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>Judge 模型</h3><h2>{report['summary']['judge_model']}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h3>RAG 平均分</h3><h2>{report['summary']['rag_avg']}</h2></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><h3>Agent 平均分</h3><h2>{report['summary']['agent_avg']}</h2></div>", unsafe_allow_html=True)

    st.subheader("系统架构")
    st.code(
        """
+------------------+
|   Test Dataset   |
+------------------+
         ↓
+------------------+
|   RAG / Agent    |
+------------------+
         ↓
+------------------+
| Evaluation Core  |
+------------------+
    ↓       ↓       ↓
RAG Eval Agent Eval LLM Judge
         ↓
+------------------+
|   Report System  |
+------------------+
"""
    )

    tab1, tab2, tab3, tab4 = st.tabs(["RAG 评分", "Agent 执行流程", "Benchmark 分析", "报告 JSON"])

    with tab1:
        fig = px.bar(
            rag_df,
            x="sample_id",
            y=["Context Relevance", "Answer Faithfulness", "Answer Correctness"],
            color_discrete_sequence=["#00ffa3", "#36cfff", "#ffb703"],
            barmode="group",
            facet_col="model",
            title="RAG 评分对比",
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e9f2ff")
        st.plotly_chart(fig, use_container_width=True)
        selected_rag = st.selectbox(
            "查看 RAG 样本详情",
            options=list(range(len(report["rag_results"]))),
            format_func=lambda i: f"{report['rag_results'][i]['model']} / {report['rag_results'][i]['sample_id']}",
        )
        rag_item = report["rag_results"][selected_rag]
        st.write("**问题：**", rag_item["question"])
        st.write("**问题类型：**", rag_item.get("question_type", "general"))
        st.write("**标准答案：**", rag_item["ground_truth"])
        st.write("**模型回答：**", rag_item["answer"])
        st.write("**检索上下文：**")
        for ctx in rag_item["retrieved_contexts"]:
            st.info(ctx)
        st.json(rag_item["judge"])

    with tab2:
        fig2 = px.line(
            agent_df,
            x="sample_id",
            y=["Tool Usage", "Task Completion", "Reasoning"],
            color_discrete_sequence=["#00ffa3", "#36cfff", "#ff4d6d"],
            markers=True,
            facet_col="model",
            title="Agent 评分对比",
        )
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e9f2ff")
        st.plotly_chart(fig2, use_container_width=True)
        selected_agent = st.selectbox(
            "查看 Agent 执行轨迹",
            options=list(range(len(report["agent_results"]))),
            format_func=lambda i: f"{report['agent_results'][i]['model']} / {report['agent_results'][i]['sample_id']}",
        )
        agent_item = report["agent_results"][selected_agent]
        st.write("**任务：**", agent_item["task"])
        for step in agent_item["trace"]:
            st.markdown(f"<div class='trace-box'>{json.dumps(step, ensure_ascii=False, indent=2)}</div>", unsafe_allow_html=True)
        st.write("**最终回答：**", agent_item["final_answer"])
        st.json(agent_item["judge"])

    with tab3:
        benchmark_summary = report.get("benchmark_summary", {})
        if benchmark_summary:
            benchmark_df = pd.DataFrame([
                {"question_type": key, **value} for key, value in benchmark_summary.items()
            ])
            fig3 = px.bar(
                benchmark_df,
                x="question_type",
                y="avg_score",
                color="question_type",
                title="细粒度 Benchmark 平均分",
                color_discrete_sequence=["#00ffa3", "#36cfff", "#ffb703", "#ff4d6d"],
            )
            fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e9f2ff", showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            st.dataframe(benchmark_df, hide_index=True, use_container_width=True)
        else:
            st.info("当前报告暂无 benchmark 分类统计。")

    with tab4:
        st.json(report)


def render_upload_page(model_names: List[str], judge_model: str) -> None:
    st.markdown("## 上传文档并自动测评")
    st.write("上传 `txt / pdf / docx` 文档后，系统会自动分块、写入 Qdrant，并支持自动生成细粒度 benchmark、自动生成标准答案、结果导出和历史记录保存。")
    uploaded_files = st.file_uploader("上传知识文档", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    auto_generate_questions = st.checkbox("自动生成细粒度 benchmark", value=True)
    per_type_count = st.slider("每类问题生成数量", min_value=1, max_value=4, value=2)
    custom_questions = st.text_area("手动问题（每行一个，可选；将归类为 general）", placeholder="这个文档的核心主题是什么？\n文档中提到了哪些关键结论？")

    if st.button("开始自动测评", use_container_width=True):
        if not uploaded_files:
            st.warning("请先上传至少一个文档。")
            return

        documents: List[str] = []
        file_names: List[str] = []
        for file in uploaded_files:
            file_names.append(file.name)
            text = extract_text_from_upload(file)
            documents.extend(chunk_text(text))

        if not documents:
            st.error("未提取到有效文本，请检查上传文档内容。")
            return

        runner = EvaluationRunner(model_names=model_names, judge_model=judge_model)
        manual_questions = [{"type": "general", "question": line.strip()} for line in custom_questions.splitlines() if line.strip()]

        if auto_generate_questions:
            with st.spinner("正在基于文档自动生成细粒度 benchmark..."):
                generated_questions = runner.generate_benchmark_from_documents(documents, per_type_count=per_type_count)
        else:
            generated_questions = []

        questions = manual_questions + generated_questions
        if not questions:
            questions = [
                {"type": "general", "question": "这批文档的核心主题是什么？"},
                {"type": "general", "question": "文档中最重要的结论或事实有哪些？"},
                {"type": "general", "question": "如果要总结给团队，应该如何概括这些文档？"},
            ]

        st.markdown("### 本次 Benchmark 问题集")
        st.json(questions)

        with st.spinner("正在自动生成标准答案..."):
            rag_samples = runner.build_uploaded_rag_dataset(documents, questions)
        with st.spinner("正在调用 Ollama 和 Qdrant 执行测评..."):
            results = runner.run_rag(rag_samples)

        report = {
            "summary": {
                "models_compared": model_names,
                "judge_model": judge_model,
                "rag_avg": round(sum(item["metrics"]["overall"] for item in results) / max(len(results), 1), 3),
                "agent_avg": 0.0,
            },
            "benchmark_summary": runner.summarize_benchmark(results),
            "rag_results": results,
            "agent_results": [],
            "generated_questions": questions,
        }
        history_path = runner.save_history(
            report,
            task_type="uploaded-documents",
            metadata={
                "files": file_names,
                "chunk_count": len(documents),
                "question_count": len(questions),
            },
        )
        st.success(f"上传文档测评完成，历史记录已保存：{history_path.name}")

        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        export_col1.download_button("导出 JSON", data=json.dumps(report, ensure_ascii=False, indent=2), file_name="evaluation-report.json", mime="application/json", use_container_width=True)
        export_col2.download_button("导出 CSV", data=pd.DataFrame(results).to_csv(index=False), file_name="evaluation-report.csv", mime="text/csv", use_container_width=True)
        export_col3.download_button("导出 Markdown", data=runner.export_report_markdown(report), file_name="evaluation-report.md", mime="text/markdown", use_container_width=True)
        export_col4.download_button("导出 HTML", data=runner.export_report_html(report), file_name="evaluation-report.html", mime="text/html", use_container_width=True)

        upload_df = pd.DataFrame([
            {"sample_id": item["sample_id"], "model": item["model"], "question_type": item.get("question_type", "general"), **item["metrics"]}
            for item in results
        ])
        fig = px.bar(
            upload_df,
            x="sample_id",
            y=["Context Relevance", "Answer Faithfulness", "Answer Correctness"],
            barmode="group",
            facet_col="model",
            color_discrete_sequence=["#00ffa3", "#36cfff", "#ffb703"],
            title="上传文档自动测评结果",
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e9f2ff")
        st.plotly_chart(fig, use_container_width=True)

        benchmark_df = pd.DataFrame([
            {"question_type": key, **value} for key, value in report["benchmark_summary"].items()
        ])
        if not benchmark_df.empty:
            st.markdown("### Benchmark 分类统计")
            st.dataframe(benchmark_df, hide_index=True, use_container_width=True)

        for item in results:
            with st.expander(f"{item['model']} / {item['sample_id']} - {item['question']}"):
                st.write("**问题类型：**", item.get("question_type", "general"))
                st.write("**标准答案：**", item["ground_truth"])
                st.write("**模型回答：**", item["answer"])
                st.write("**检索上下文：**")
                for ctx in item["retrieved_contexts"]:
                    st.info(ctx)
                st.json(item["metrics"])
                st.json(item["judge"])


def render_history_page(model_names: List[str], judge_model: str) -> None:
    st.markdown("## 历史评测记录")
    runner = EvaluationRunner(model_names=model_names, judge_model=judge_model)
    history_items = runner.list_history()
    if not history_items:
        st.info("暂无历史评测记录。")
        return

    task_types = sorted({item["task_type"] for item in history_items})
    model_filters = sorted({model for item in history_items for model in item.get("models", [])})

    col1, col2, col3 = st.columns(3)
    selected_task_types = col1.multiselect("按任务类型筛选", options=task_types, default=task_types)
    selected_models_filter = col2.multiselect("按模型筛选", options=model_filters, default=model_filters)
    keyword = col3.text_input("关键词搜索", placeholder="文件名 / 模型 / Judge / 任务类型")

    keyword_lower = keyword.strip().lower()
    filtered_items = []
    for item in history_items:
        matches_filter = item["task_type"] in selected_task_types and (
            not selected_models_filter or any(model in selected_models_filter for model in item.get("models", []))
        )
        if not matches_filter:
            continue
        haystack = " ".join([
            item.get("file_name", ""),
            item.get("task_type", ""),
            item.get("judge_model", ""),
            " ".join(item.get("models", [])),
        ]).lower()
        if keyword_lower and keyword_lower not in haystack:
            continue
        filtered_items.append(item)

    if not filtered_items:
        st.warning("筛选后没有匹配的历史记录。")
        return

    history_df = pd.DataFrame(filtered_items)
    st.dataframe(history_df, hide_index=True, use_container_width=True)

    bulk_col1, bulk_col2 = st.columns(2)
    selected_for_delete = bulk_col1.multiselect("批量删除历史记录", options=[item["file_name"] for item in filtered_items])
    if bulk_col1.button("批量删除选中记录", use_container_width=True):
        deleted_count = runner.delete_histories(selected_for_delete)
        st.success(f"已批量删除 {deleted_count} 条历史记录。")
        st.rerun()
    if bulk_col2.button("清空全部历史记录", use_container_width=True):
        deleted_count = runner.clear_history()
        st.success(f"已清空 {deleted_count} 条历史记录。")
        st.rerun()

    selected_file = st.selectbox("选择历史记录", options=[item["file_name"] for item in filtered_items])
    payload = runner.load_history(selected_file)

    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    action_col1.download_button("导出 JSON", data=json.dumps(payload, ensure_ascii=False, indent=2), file_name=selected_file, mime="application/json", use_container_width=True)
    action_col2.download_button("导出 Markdown", data=runner.export_report_markdown(payload), file_name=selected_file.replace(".json", ".md"), mime="text/markdown", use_container_width=True)
    action_col3.download_button("导出 HTML", data=runner.export_report_html(payload), file_name=selected_file.replace(".json", ".html"), mime="text/html", use_container_width=True)
    if action_col4.button("删除该记录", use_container_width=True):
        runner.delete_history(selected_file)
        st.success(f"已删除历史记录：{selected_file}")
        st.rerun()

    st.write("**任务类型：**", payload.get("task_type", ""))
    st.write("**创建时间：**", payload.get("created_at", ""))
    st.write("**元数据：**")
    st.json(payload.get("metadata", {}))
    st.write("**完整报告：**")
    st.json(payload.get("report", {}))


st.sidebar.title("运行状态")
st.sidebar.write("### 模型配置")
available_models = DEFAULT_MODELS or ["qwen3:8b"]
selected_models = st.sidebar.multiselect("选择参评模型", options=available_models, default=[available_models[0]])
if not selected_models:
    selected_models = [available_models[0]]
judge_model = st.sidebar.selectbox("选择 Judge / 标准答案模型", options=available_models, index=available_models.index(DEFAULT_JUDGE_MODEL) if DEFAULT_JUDGE_MODEL in available_models else 0)

status_rows = []
for model_name in available_models:
    client = OllamaClient(model_name)
    status_rows.append({"model": model_name, "available": "是" if client.is_available() else "否"})
st.sidebar.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)

page = st.sidebar.radio("页面", ["平台总览", "上传文档自动测评", "历史评测记录"])

if page == "平台总览":
    report = run_demo(selected_models, judge_model)
    render_dashboard(report)
elif page == "上传文档自动测评":
    render_upload_page(selected_models, judge_model)
else:
    render_history_page(selected_models, judge_model)
