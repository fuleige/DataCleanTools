from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        return Path(uri)
    raise ValueError(f"Only local file URIs are supported by this implementation: {uri}")


def path_to_file_uri(path: str | Path) -> str:
    return Path(path).expanduser().resolve().as_uri()


def ensure_local_uri(value: str | Path) -> str:
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme:
        if parsed.scheme != "file":
            return text
        return Path(unquote(parsed.path)).expanduser().resolve().as_uri()
    return Path(text).expanduser().resolve().as_uri()


def is_local_uri(uri: str) -> bool:
    return urlparse(uri).scheme in ("", "file")
