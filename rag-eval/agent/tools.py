from __future__ import annotations

from typing import Callable, Dict


def weather(city: str) -> Dict:
    weather_map = {
        "北京": {"condition": "晴", "temperature": 24, "advice": "适合外出，注意补水。"},
        "上海": {"condition": "多云", "temperature": 22, "advice": "可正常出行，建议携带薄外套。"},
    }
    return weather_map.get(city, {"condition": "未知", "temperature": 20, "advice": "请结合实时天气决定。"})


def fx_rate(base: str, quote: str) -> Dict:
    rates = {("USD", "CNY"): 7.12, ("EUR", "CNY"): 7.83}
    return {"rate": rates.get((base, quote), 1.0), "base": base, "quote": quote}


def project_status(project: str) -> Dict:
    return {
        "project": project,
        "status": "进行中",
        "milestone": "已完成 RAG 评估闭环，正在联调 Agent 可视化。",
    }


TOOL_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "weather": {"handler": weather, "description": "根据城市查询天气与出行建议"},
    "fx_rate": {"handler": fx_rate, "description": "查询汇率并用于金额换算"},
    "project_status": {"handler": project_status, "description": "读取项目当前状态和里程碑信息"},
}
