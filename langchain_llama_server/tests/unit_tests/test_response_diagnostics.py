from copy import deepcopy

import pytest
from langchain_core.messages import AIMessageChunk
from openai.types.chat import ChatCompletion

from langchain_llama_server.chat_models import ChatLlamaServer


DIAGNOSTICS = {
    "metrics": {"speculative_decoding": {"mean_acceptance_length": 4.2}},
    "timings": {"choices": [{"decode_time": 0.25}]},
    "sglext": {"spec_tokens_details": {"spec_accept_rate": 0.8}},
}


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("quiet", [False, True])
@pytest.mark.parametrize("extensions", [DIAGNOSTICS, {}, dict.fromkeys(DIAGNOSTICS)])
def test_preserves_optional_diagnostics(streaming, quiet, extensions):
    model = ChatLlamaServer(api_key="none", model="test", quiet=quiet)
    response = {
        "id": "test",
        "created": 1,
        "model": "test",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        **deepcopy(extensions),
    }
    if streaming:
        response.update(object="chat.completion.chunk", choices=[])
        generation = model._convert_chunk_to_generation_chunk(
            response, AIMessageChunk, None
        )
        # Exercise aggregation of the usage-only tail with a content chunk.
        message = AIMessageChunk(content="hello") + generation.message
    else:
        response.update(
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
        )
        message = (
            model._create_chat_result(ChatCompletion.model_validate(response))
            .generations[0]
            .message
        )

    assert message.usage_metadata["total_tokens"] == 12
    for field, value in extensions.items():
        if value is None:
            assert field not in message.response_metadata
        else:
            assert message.response_metadata[field] == value
            response[field].clear()
            assert message.response_metadata[field] == value
        assert field not in message.additional_kwargs
