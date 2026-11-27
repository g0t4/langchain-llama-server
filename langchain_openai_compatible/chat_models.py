import re
from collections.abc import AsyncIterator, Iterator
from copy import deepcopy
from typing import Any

import openai
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai.chat_models.base import BaseChatOpenAI


def _preserve_reasoning(message: BaseMessage, fields: dict) -> None:
    reasoning = fields.get("reasoning_content")
    if reasoning is None:
        reasoning = fields.get("reasoning")
    if reasoning is not None:
        message.additional_kwargs["reasoning_content"] = reasoning


def _infer_backend(payload: dict) -> str | None:
    fingerprint = payload.get("system_fingerprint") or ""
    if fingerprint.startswith("vllm-"):
        return "vllm"
    if re.fullmatch(r"b[0-9]+-[0-9a-f]+", fingerprint):
        return "llama-server"

    timings = payload.get("timings")
    if payload.get("sglext") is not None or (
        isinstance(timings, dict) and "choices" in timings
    ):
        return "sglang"
    if isinstance(timings, dict):
        return "llama-server"
    if isinstance(payload.get("usage"), dict) and isinstance(payload.get("metrics"), dict):
        return "vllm"
    if any("matched_stop" in choice for choice in payload.get("choices") or []):
        return "sglang"
    return None


def _label_stream_chunk(chunk: ChatGenerationChunk) -> bool:
    metadata = chunk.message.response_metadata
    for payload in metadata.get("raw_chunks", []):
        if backend := _infer_backend(payload):
            metadata["backend"] = backend
            return True
    return False


class ChatOpenAICompatible(BaseChatOpenAI):
    """OpenAI-compatible chat completions with the full parsed backend payload."""

    # LangChain concatenates custom string metadata when adding chunks.
    # Emit the label only on the first identifiable event of each request.
    def _stream(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        labeled = False
        for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            labeled = labeled or _label_stream_chunk(chunk)
            yield chunk

    async def _astream(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        labeled = False
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            labeled = labeled or _label_stream_chunk(chunk)
            yield chunk

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None:
            return None

        message = generation.message
        # LangChain concatenates lists when callers add message chunks together.
        message.response_metadata["raw_chunks"] = [deepcopy(chunk)]
        message.response_metadata["model_provider"] = "openai_compatible"
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices")
        if choices:
            _preserve_reasoning(message, choices[0].get("delta") or {})
        return generation

    def _create_chat_result(
        self, response: dict | openai.BaseModel, generation_info: dict | None = None
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        payload = response if isinstance(response, dict) else response.model_dump()
        backend = _infer_backend(payload)
        for generation, choice in zip(result.generations, payload["choices"]):
            generation.message.response_metadata["raw_response"] = deepcopy(payload)
            if backend:
                generation.message.response_metadata["backend"] = backend
            _preserve_reasoning(generation.message, choice["message"])
        if result.llm_output is not None:
            result.llm_output["model_provider"] = "openai_compatible"
        return result
