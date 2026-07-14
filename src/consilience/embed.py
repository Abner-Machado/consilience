"""Local embeddings via Ollama.

We talk to Ollama over plain HTTP so there is no cloud call and no API key. If
Ollama is not running, or the model is not pulled, we fail with a message that
tells the reader exactly what to do instead of a stack trace.
"""

from __future__ import annotations

import httpx
import numpy as np

from .config import Config


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or the model is unavailable."""


class Embedder:
    def __init__(self, config: Config):
        self._config = config
        self._client = httpx.Client(base_url=config.ollama_host, timeout=120.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Embedder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _embed_one(self, text: str) -> list[float]:
        try:
            response = self._client.post(
                "/api/embeddings",
                json={"model": self._config.model, "prompt": text},
            )
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self._config.ollama_host}. "
                "Start it with `ollama serve`."
            ) from exc

        if response.status_code >= 400:
            try:
                detail = response.json().get("error", "") or ""
            except ValueError:
                detail = response.text[:200]
            if response.status_code == 404 or "not found" in detail.lower():
                raise OllamaError(
                    f"Model '{self._config.model}' is not available. "
                    f"Pull it with `ollama pull {self._config.model}`."
                )
            raise OllamaError(
                f"Ollama could not embed with model '{self._config.model}' "
                f"(HTTP {response.status_code}). {detail}".strip()
            )

        vector = response.json().get("embedding")
        if not vector:
            raise OllamaError(
                f"Ollama returned no embedding for model '{self._config.model}'."
            )
        return vector

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a (n, dim) float32 matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = [self._embed_one(text) for text in texts]
        return np.asarray(vectors, dtype=np.float32)
