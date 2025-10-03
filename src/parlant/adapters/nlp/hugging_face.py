# Copyright 2025 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any
from typing_extensions import override
import torch  # type: ignore
from transformers import AutoModel, AutoTokenizer  # type: ignore
from huggingface_hub.errors import (  # type: ignore
    InferenceTimeoutError,
    InferenceEndpointError,
    InferenceEndpointTimeoutError,
    TextGenerationError,
)

from tempfile import gettempdir

from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.embedding import Embedder, EmbeddingResult


_TOKENIZER_MODELS: dict[str, AutoTokenizer] = {}
_AUTO_MODELS: dict[str, AutoModel] = {}
_DEVICE: torch.device | None = None


def _model_temp_dir() -> str:
    return str(Path(gettempdir()) / "parlant_data" / "hf_models")


def _create_tokenizer_sync(model_name: str) -> AutoTokenizer:
    """Synchronous tokenizer creation - should only be called in thread pool."""
    if model_name in _TOKENIZER_MODELS:
        return _TOKENIZER_MODELS[model_name]

    save_dir = os.environ.get("PARLANT_HOME", _model_temp_dir())
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)

    _TOKENIZER_MODELS[model_name] = tokenizer

    return tokenizer


async def _create_tokenizer(model_name: str) -> AutoTokenizer:
    """Async-safe tokenizer creation using thread pool to avoid blocking."""
    return await asyncio.to_thread(_create_tokenizer_sync, model_name)


def _get_device() -> torch.device:
    global _DEVICE

    if _DEVICE:
        return _DEVICE

    if torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    else:
        _DEVICE = torch.device("cpu")

    return _DEVICE


def _create_auto_model_sync(model_name: str) -> AutoModel:
    """Synchronous model creation - should only be called in thread pool."""
    if model_name in _AUTO_MODELS:
        return _AUTO_MODELS[model_name]

    save_dir = os.environ.get("PARLANT_HOME", _model_temp_dir())
    os.makedirs(save_dir, exist_ok=True)

    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        attn_implementation="eager",
    ).to(_get_device())

    model.save_pretrained(save_dir)
    model.eval()

    _AUTO_MODELS[model_name] = model

    return model


async def _create_auto_model(model_name: str) -> AutoModel:
    """Async-safe model creation using thread pool to avoid blocking."""
    return await asyncio.to_thread(_create_auto_model_sync, model_name)


class HuggingFaceEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._tokenizer: AutoTokenizer | None = None
        self._tokenizer_initialized = False

    async def _ensure_tokenizer(self) -> None:
        """Ensure tokenizer is initialized asynchronously."""
        if not self._tokenizer_initialized:
            self._tokenizer = await _create_tokenizer(self.model_name)
            self._tokenizer_initialized = True

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        await self._ensure_tokenizer()
        assert self._tokenizer is not None
        tokens = self._tokenizer.tokenize(prompt)
        return len(tokens)


class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: AutoModel | None = None
        self._model_initialized = False
        self._tokenizer = HuggingFaceEstimatingTokenizer(model_name=model_name)

    async def _ensure_model(self) -> None:
        """Ensure model is initialized asynchronously."""
        if not self._model_initialized:
            self._model = await _create_auto_model(self.model_name)
            self._model_initialized = True

    @property
    @override
    def id(self) -> str:
        return f"hugging-face/{self.model_name}"

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def tokenizer(self) -> HuggingFaceEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def dimensions(self) -> int:
        # Default embedding dimension for most BERT-style models
        return 768

    @policy(
        [
            retry(
                exceptions=(
                    InferenceTimeoutError,
                    InferenceEndpointError,
                    InferenceEndpointTimeoutError,
                ),
                max_exceptions=2,
            ),
            retry(exceptions=(TextGenerationError), max_exceptions=3),
        ]
    )
    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        # Ensure both model and tokenizer are initialized
        await self._ensure_model()
        await self._tokenizer._ensure_tokenizer()

        assert self._model is not None
        assert self._tokenizer._tokenizer is not None

        tokenized_texts = self._tokenizer._tokenizer.batch_encode_plus(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized_texts = {key: value.to(_get_device()) for key, value in tokenized_texts.items()}

        with torch.no_grad():
            embeddings = self._model(**tokenized_texts).last_hidden_state[:, 0, :]

        return EmbeddingResult(vectors=embeddings.tolist())


class JinaAIEmbedder(HuggingFaceEmbedder):
    def __init__(self) -> None:
        super().__init__("jinaai/jina-embeddings-v2-base-en")

    @property
    @override
    def dimensions(self) -> int:
        return 768
