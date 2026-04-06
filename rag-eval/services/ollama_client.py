from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests


class OllamaClient:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def generate(self, prompt: str, format: str | None = None) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if format:
            payload["format"] = format
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def embed(self, text: str) -> List[float]:
        payload = {
            "model": self.model_name,
            "input": text,
        }
        response = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError("Ollama 未返回 embedding 结果")
        return embeddings[0]

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(model.get("name") == self.model_name for model in models)
        except requests.RequestException:
            return False

