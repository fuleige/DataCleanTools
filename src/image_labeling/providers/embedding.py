from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def build_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    provider = config.provider.lower()
    if provider in {"simple_color", "simple-color", "simple_color_histogram"}:
        return SimpleColorEmbeddingProvider(
            name=config.model_name,
            version=config.model_version,
            normalize=config.normalize,
        )
    raise EmbeddingProviderError(
        f"Embedding provider '{config.provider}' is declared but not implemented yet. "
        "Use provider=simple_color for local MVP runs, or add a provider adapter."
    )
