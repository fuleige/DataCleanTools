from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from image_labeling.config import LabelConfig, PrelabelConfig


class PrelabelProviderError(RuntimeError):
    pass


class PrelabelProvider:
    provider_name: str
    model_version: str
    prompt_version: str | None

    def predict(self, rows: list[dict[str, Any]], labels: list[LabelConfig], top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError


def _label_candidates(labels: list[LabelConfig]) -> list[LabelConfig]:
    return labels or [LabelConfig(id="unknown", name="Unknown")]


@dataclass
class MockPrelabelProvider(PrelabelProvider):
    provider_name: str = "mock"
    model_version: str = "mock-v1"
    prompt_version: str | None = None

    def predict(self, rows: list[dict[str, Any]], labels: list[LabelConfig], top_k: int) -> list[dict[str, Any]]:
        label_items = _label_candidates(labels)
        outputs = []
        for row in rows:
            weak_label = (row.get("metadata") or {}).get("label") or (row.get("metadata") or {}).get("weak_label")
            ordered = list(label_items)
            if weak_label:
                ordered.sort(key=lambda item: item.id != weak_label)
            else:
                rng = random.Random(row["image_id"])
                rng.shuffle(ordered)
            candidates = []
            for rank, label in enumerate(ordered[: max(1, top_k)], start=1):
                if rank == 1 and weak_label == label.id:
                    confidence = 0.96
                elif rank == 1:
                    confidence = 0.55 + (abs(hash(row["image_id"])) % 40) / 100
                else:
                    confidence = max(0.01, 0.45 - rank * 0.08)
                candidates.append({"label": label.id, "confidence": round(float(confidence), 4), "rank": rank})
            outputs.append(
                {
                    "image_id": row["image_id"],
                    "provider": self.provider_name,
                    "model_version": self.model_version,
                    "prompt_version": self.prompt_version,
                    "candidates": candidates,
                    "status": "succeeded",
                    "error_code": None,
                }
            )
        return outputs


@dataclass
class OpenAICompatiblePrelabelProvider(PrelabelProvider):
    provider_name: str
    model_version: str
    prompt_version: str | None
    base_url: str
    api_key: str
    timeout_seconds: int
    max_retries: int
    rate_limit_qps: float

    def predict(self, rows: list[dict[str, Any]], labels: list[LabelConfig], top_k: int) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        min_interval = 1.0 / self.rate_limit_qps if self.rate_limit_qps > 0 else 0
        last_call = 0.0
        for row in rows:
            elapsed = time.time() - last_call
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_call = time.time()
            outputs.append(self._predict_one(row, labels, top_k))
        return outputs

    def _predict_one(self, row: dict[str, Any], labels: list[LabelConfig], top_k: int) -> dict[str, Any]:
        image_path = Path(row["local_path"])
        data_url = _image_data_url(image_path)
        label_text = "\n".join(f"- {label.id}: {label.name} {label.description}".strip() for label in labels)
        prompt = (
            "You are labeling a single image for a single-label classification task. "
            "Return strict JSON only with a candidates array. "
            f"Allowed labels:\n{label_text}\n"
            f"Return up to {top_k} candidates as objects with label, confidence, and rank."
        )
        payload = {
            "model": self.model_version,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        error_code = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(self.base_url.rstrip("/") + "/chat/completions", json=payload, headers=headers)
                    response.raise_for_status()
                parsed = response.json()
                content = parsed["choices"][0]["message"]["content"]
                candidates = _parse_candidates(content, labels, top_k)
                return {
                    "image_id": row["image_id"],
                    "provider": self.provider_name,
                    "model_version": self.model_version,
                    "prompt_version": self.prompt_version,
                    "candidates": candidates,
                    "status": "succeeded",
                    "error_code": None,
                }
            except Exception as exc:  # noqa: BLE001
                error_code = exc.__class__.__name__
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        return {
            "image_id": row["image_id"],
            "provider": self.provider_name,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "candidates": [],
            "status": "failed",
            "error_code": error_code,
        }


def _image_data_url(path: Path) -> str:
    mime = "image/jpeg"
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    with path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_candidates(content: str, labels: list[LabelConfig], top_k: int) -> list[dict[str, Any]]:
    label_ids = {label.id for label in labels}
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    candidates = payload.get("candidates", payload if isinstance(payload, list) else [])
    normalized = []
    for idx, candidate in enumerate(candidates[:top_k], start=1):
        label = candidate.get("label")
        if label not in label_ids:
            continue
        confidence = float(candidate.get("confidence", 0.0))
        normalized.append({"label": label, "confidence": max(0.0, min(1.0, confidence)), "rank": idx})
    return normalized


def build_prelabel_provider(config: PrelabelConfig) -> PrelabelProvider:
    provider = config.provider.lower()
    provider_config = config.provider_config
    if provider in {"mock", "dev_mock"} or provider_config.get("mock"):
        return MockPrelabelProvider(
            provider_name=provider,
            model_version=str(provider_config.get("model", "mock-v1")),
            prompt_version=config.prompt_version,
        )
    if provider == "multimodal_api":
        api_key_env = provider_config.get("api_key_env", "MULTIMODAL_API_KEY")
        api_key = os.environ.get(api_key_env)
        model = provider_config.get("model")
        base_url = provider_config.get("base_url", os.environ.get("MULTIMODAL_API_BASE_URL"))
        if not api_key or not model or not base_url:
            raise PrelabelProviderError(
                "multimodal_api requires provider_config.model, provider_config.base_url, "
                f"and API key env {api_key_env}; set provider_config.mock=true for local MVP runs."
            )
        return OpenAICompatiblePrelabelProvider(
            provider_name="multimodal_api",
            model_version=str(model),
            prompt_version=config.prompt_version,
            base_url=str(base_url),
            api_key=api_key,
            timeout_seconds=int(provider_config.get("timeout_seconds", 60)),
            max_retries=int(provider_config.get("max_retries", 3)),
            rate_limit_qps=float(provider_config.get("rate_limit_qps", 2)),
        )
    raise PrelabelProviderError(f"Prelabel provider '{config.provider}' is not implemented yet.")
