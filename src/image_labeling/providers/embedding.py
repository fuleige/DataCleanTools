from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_labeling.config import EmbeddingConfig


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingProvider:
    name: str
    version: str

    def encode(self, image_paths: list[Path]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SimpleColorEmbeddingProvider(EmbeddingProvider):
    name: str
    version: str
    normalize: bool = True

    def encode(self, image_paths: list[Path]) -> np.ndarray:
        vectors = [self._encode_one(path) for path in image_paths]
        matrix = np.asarray(vectors, dtype=np.float32)
        if self.normalize and len(matrix):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        return matrix

    def _encode_one(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((96, 96))
            arr = np.asarray(rgb, dtype=np.float32)
        channels = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0, 255), density=True)
            channels.append(hist.astype(np.float32))
        gray = np.mean(arr, axis=2)
        gray_hist, _ = np.histogram(gray, bins=16, range=(0, 255), density=True)
        features = np.concatenate([*channels, gray_hist.astype(np.float32)])
        return features


class SiglipEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise EmbeddingProviderError(
                "SigLIP embedding requires optional deep learning dependencies. "
                "Install them with: env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy "
                "-u ALL_PROXY -u all_proxy /data/envs/dataclean-tools/bin/python -m pip install "
                "\"dataclean-tools[deep]\""
            ) from exc

        self._torch = torch
        provider_config = config.provider_config or {}
        model_path = str(
            provider_config.get("model_path")
            or provider_config.get("pretrained_model_name_or_path")
            or config.model_name
        )
        local_files_only = bool(provider_config.get("local_files_only", Path(model_path).exists()))
        trust_remote_code = bool(provider_config.get("trust_remote_code", False))
        device = str(provider_config.get("device", "auto"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = self._resolve_dtype(str(provider_config.get("dtype", "auto")), device)

        self.name = config.model_name
        self.version = config.model_version
        self.normalize = config.normalize
        self.device = torch.device(device)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        try:
            self.model = AutoModel.from_pretrained(model_path, dtype=dtype, **model_kwargs)
        except TypeError:
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_dtype(self, dtype: str, device: str) -> Any:
        torch = self._torch
        normalized = dtype.lower()
        if normalized == "auto":
            return torch.float16 if device.startswith("cuda") else torch.float32
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float32", "fp32"}:
            return torch.float32
        raise EmbeddingProviderError("embedding.provider_config.dtype must be one of auto, float16, bfloat16, float32")

    def encode(self, image_paths: list[Path]) -> np.ndarray:
        if not image_paths:
            return np.zeros((0, 0), dtype=np.float32)

        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())

        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}

        torch = self._torch
        with torch.inference_mode():
            if hasattr(self.model, "get_image_features"):
                features = self._feature_tensor(self.model.get_image_features(**inputs))
            else:
                outputs = self.model.vision_model(**inputs)
                features = self._feature_tensor(outputs)
                if hasattr(self.model, "visual_projection"):
                    features = self.model.visual_projection(features)
            features = features.float()
            if self.normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def _feature_tensor(self, output: Any) -> Any:
        if hasattr(output, "float"):
            return output
        pooler_output = getattr(output, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        last_hidden_state = getattr(output, "last_hidden_state", None)
        if last_hidden_state is not None:
            return last_hidden_state[:, 0]
        raise EmbeddingProviderError("SigLIP model output does not contain image features")


def build_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    provider = config.provider.lower()
    if provider in {"simple_color", "simple-color", "simple_color_histogram"}:
        return SimpleColorEmbeddingProvider(
            name=config.model_name,
            version=config.model_version,
            normalize=config.normalize,
        )
    if provider in {"siglip", "siglip2", "hf_siglip", "hf-siglip"}:
        return SiglipEmbeddingProvider(config)
    raise EmbeddingProviderError(
        f"Embedding provider '{config.provider}' is declared but not implemented yet. "
        "Use provider=simple_color for local MVP runs, provider=siglip2 for Hugging Face SigLIP/SigLIP2 models, "
        "or add a provider adapter."
    )
