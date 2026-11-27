import pytest
from langchain_core.messages import AIMessageChunk
from openai.types.chat import ChatCompletion

from langchain_llama_server.chat_models import ChatLlamaServer


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"reasoning": "thought"}, "thought"),
        ({"reasoning_content": "thought"}, "thought"),
        ({"reasoning_content": None, "reasoning": "fallback"}, "fallback"),
        ({"reasoning_content": "preferred", "reasoning": "other"}, "preferred"),
        ({"reasoning": ""}, ""),
        ({"reasoning": None}, None),
        ({}, None),
    ],
)
def test_reasoning_field_normalization(streaming, fields, expected):
    model = ChatLlamaServer(api_key="none", model="test")
    response = {"id": "test", "created": 1, "model": "test"}
    if streaming:
        response.update(
            object="chat.completion.chunk",
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", **fields},
                    "finish_reason": None,
                }
            ],
        )
        message = model._convert_chunk_to_generation_chunk(
            response, AIMessageChunk, None
        ).message
    else:
        response.update(
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer", **fields},
                    "finish_reason": "stop",
                }
            ],
        )
        message = (
            model._create_chat_result(ChatCompletion.model_validate(response))
            .generations[0]
            .message
        )
    if expected is None:
        assert "reasoning_content" not in message.additional_kwargs
    else:
        assert message.additional_kwargs["reasoning_content"] == expected
