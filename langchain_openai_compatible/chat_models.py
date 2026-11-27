from copy import deepcopy

import openai
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai.chat_models.base import BaseChatOpenAI


def _preserve_reasoning(message: BaseMessage, fields: dict) -> None:
    reasoning = fields.get("reasoning_content")
    if reasoning is None:
        reasoning = fields.get("reasoning")
    if reasoning is not None:
        message.additional_kwargs["reasoning_content"] = reasoning


class ChatOpenAICompatible(BaseChatOpenAI):
    """OpenAI-compatible chat completions with the full parsed backend payload."""

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
        for generation, choice in zip(result.generations, payload["choices"]):
            generation.message.response_metadata["raw_response"] = deepcopy(payload)
            _preserve_reasoning(generation.message, choice["message"])
        if result.llm_output is not None:
            result.llm_output["model_provider"] = "openai_compatible"
        return result
