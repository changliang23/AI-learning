# RAG & Agent Evaluation Platform

一个基于本地 `Ollama + Qdrant` 的 AI 应用测评 Demo，用于评估 `RAG` 与 `Agent` 系统表现，并通过 `Streamlit` 展示评分结果、执行轨迹、上传文档自动测评、多模型横向对比、细粒度 benchmark 自动生成、结果导出以及历史评测记录管理。

## 功能特性

- 使用本地 `Ollama` 大模型接口替代模拟输出
- 使用 `Qdrant` 作为向量数据库进行文档检索
- 支持 RAG 测评维度：
  - `Context Relevance`
  - `Answer Faithfulness`
  - `Answer Correctness`
- 支持 Agent 测评维度：
  - `Tool Usage`
  - `Task Completion`
  - `Reasoning`
- 支持上传 `txt / pdf / docx` 文档后自动分块、检索与测评
- 支持对上传文档自动生成标准答案，再进行 Correctness 评分
- 支持自动生成细粒度 benchmark 问题集
- 支持按问题类型统计 benchmark 分数
- 支持多模型横向对比与独立 Judge 模型选择
- 支持评测任务保存、筛选、单条删除、批量删除、清空与历史记录查看
- 支持将评测结果导出为 `JSON / CSV / Markdown / 单文件自包含 HTML`
- 支持生成 `reports/report.json` 与 `reports/history/*.json`

## 项目结构

```text
rag-eval/
│
├── datasets/
│   ├── rag.json
│   └── agent.json
├── rag/
│   ├── retriever.py
│   └── generator.py
├── agent/
│   ├── agent.py
│   └── tools.py
├── evaluators/
│   ├── rag_eval.py
│   ├── agent_eval.py
│   └── llm_judge.py
├── core/
│   └── runner.py
├── services/
│   └── ollama_client.py
├── reports/
│   ├── report.json
│   └── history/
├── main.py
└── requirements.txt
```

## 环境准备

### 1. 启动 Ollama

确保本机已安装 Ollama，并拉取你希望用于对比的模型，例如：

```bash
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
```

如果你打算单独使用 embedding 模型，也可以额外拉取：

```bash
ollama pull nomic-embed-text
```

默认访问地址：

```bash
http://localhost:11434
```

### 2. 启动 Qdrant

推荐使用 Docker：

```bash
docker run -p 6333:6333 qdrant/qdrant
```

默认访问地址：

```bash
http://localhost:6333
```

### 3. 安装依赖

```bash
cd /Users/liangchang/code/AI-learning/rag-eval
python3 -m pip install -r requirements.txt
```

## 启动项目

```bash
streamlit run main.py
```

## 可选环境变量

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export QDRANT_URL=http://localhost:6333
export OLLAMA_EMBED_MODEL=nomic-embed-text
export AVAILABLE_OLLAMA_MODELS=qwen3:8b,llama3.1:8b,deepseek-r1:8b
export JUDGE_MODEL=qwen3:8b
```

## 页面说明

### 平台总览

- 支持在侧边栏多选参评模型
- 支持单独指定 Judge / 标准答案生成模型
- 展示 RAG 多维度评分图表
- 展示 Agent 执行流程和执行轨迹
- 展示细粒度 benchmark 分类统计
- 展示自动生成报告 JSON
- 自动保存一次平台评测结果到历史记录

### 上传文档自动测评

- 上传一个或多个 `txt / pdf / docx`
- 自动抽取文本并切分 chunk
- 自动写入 Qdrant
- 可开启“自动生成细粒度 benchmark”
- 可设置每类 benchmark 问题数量
- 也可手动输入评测问题，手动问题默认类型为 `general`
- 使用 Judge 模型先根据文档生成标准答案
- 使用多个候选模型分别进行检索增强问答
- 输出评分图表、标准答案、模型回答、上下文和 judge 结果
- 输出 benchmark 分类统计
- 支持结果导出为 `JSON / CSV / Markdown / HTML`
- 自动保存本次评测任务到历史记录

### 历史评测记录

- 浏览保存在 `reports/history/` 下的所有历史任务
- 支持按任务类型筛选
- 支持按模型筛选
- 支持按关键词搜索历史记录
- 查看任务类型、创建时间、模型组合、Judge 模型
- 查看上传任务的文件名、chunk 数量、问题数量等元数据
- 支持导出历史记录为 `JSON / Markdown / HTML`
- 支持按关键词搜索历史记录
- 支持删除单条历史记录
- 支持批量删除选中历史记录
- 支持清空全部历史记录
- 回看完整报告 JSON

## 说明

### 细粒度 benchmark 自动生成

在上传文档页面中，开启“自动生成细粒度 benchmark”后，系统会调用 Judge 模型基于文档内容生成多类型问题集。当前默认支持：

- `factual`
- `summary`
- `comparison`
- `reasoning`

每类问题数量可以在 UI 中配置。生成后的问题集会直接用于：

- 标准答案生成
- RAG 评测
- benchmark 分类统计

### Benchmark 分类统计

系统会根据问题类型对测评结果做聚合，输出每种类型的：

- `count`
- `avg_score`

你可以在平台总览页和上传文档测评页中查看这些细粒度统计结果。

### 结果导出

当前支持导出：

- `JSON`
- `CSV`
- `Markdown`
- `HTML`

其中：

- 上传文档测评页支持导出当前运行结果
- 历史记录页支持导出选中的历史记录
- `HTML` 导出会生成一个更适合汇报展示的可视化风格静态报告页
- HTML 报告为单文件自包含格式，内嵌样式与数据，可离线直接打开或发送给他人
- HTML 报告中包含模型对比雷达图、RAG 三维指标横向条形图、Agent 执行评分图、benchmark 柱状图、问题类型占比环形图和结果表格

### 历史记录管理

系统会自动把每次平台总览测评和上传文档测评保存到：

```text
reports/history/
```

每条记录都会包含：

- 创建时间
- 任务类型
- 参评模型
- Judge 模型
- 元数据
- benchmark 分类统计
- 完整报告内容

在历史记录页面中，你还可以：

- 按任务类型筛选
- 按模型筛选
- 按关键词搜索文件名、模型名、Judge 模型或任务类型
- 删除单条记录
- 批量删除多条记录
- 一键清空全部历史记录

## 注意事项

1. 当前 Agent 工具本身仍是 demo 工具，但工具选择、规划与最终回答已经由本地模型生成。
2. 如果某些模型未在 Ollama 中拉取完成，侧边栏状态会显示不可用。
3. 如果 `qwen3:8b` 不适合 embedding，请务必通过 `OLLAMA_EMBED_MODEL` 指定专用 embedding 模型，例如 `nomic-embed-text`。
4. 如果 Ollama 或 Qdrant 未启动，页面中的运行结果会失败，请先确认服务状态。
5. 自动生成 benchmark 问题集和标准答案都会调用本地模型，文档较长时耗时会增加。
