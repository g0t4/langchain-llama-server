import json
from functools import reduce
from operator import add

import httpx
from openai.types.chat import ChatCompletionChunk

from langchain_openai_compatible import ChatOpenAICompatible


def client_for(handler):
    return ChatOpenAICompatible(
        base_url="http://backend.test/v1",
        model="test",
        api_key="",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_stream_retains_backend_events_after_finish_and_repeated_usage():
    usage = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    events = [
        {"choices": [{"index": 0, "delta": {"reasoning": "think", "content": "hi"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": usage, "sglext": {"accepted": 1}},
        {"choices": [], "usage": usage, "timings": {"decode": 0.1}},
    ]
    events = [
        dict(event, id="test", object="chat.completion.chunk", model="test", created=1)
        for event in events
    ]

    def respond(request):
        assert json.loads(request.content)["stream_options"]["include_usage"]
        frames = [f"data: {json.dumps(event)}\n\n" for event in events]
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"},
            text="".join(frames) + "data: [DONE]\n\n",
        )

    chunks = list(client_for(respond).stream("Hello", stream_usage=True))
    assert [chunk.chunk_position for chunk in chunks] == [None] * 4 + ["last"]
    assert "raw_chunks" not in chunks[-1].response_metadata
    captured = [chunk.response_metadata["raw_chunks"][0] for chunk in chunks[:-1]]
    assert captured == [ChatCompletionChunk.model_validate(event).model_dump() for event in events]
    merged = reduce(add, chunks)
    assert merged.response_metadata["raw_chunks"] == captured
    assert merged.additional_kwargs["reasoning_content"] == "think"
    assert merged.content == "hi"


def test_invoke_preserves_full_response_and_keeps_it_out_of_next_request():
    payload = {
        "id": "test", "model": "test", "created": 1, "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi",
                                             "reasoning_content": "think"},
                     "finish_reason": "stop"}],
        "custom_metric": {"nested": [1, 2]},
    }
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    model = client_for(respond)
    message = model.invoke("Hello")
    raw = message.response_metadata["raw_response"]
    assert raw["choices"][0]["message"]["reasoning_content"] == "think"
    assert raw["custom_metric"] == payload["custom_metric"]
    assert message.response_metadata["model_provider"] == "openai_compatible"
    assert message.additional_kwargs["reasoning_content"] == "think"
    model.invoke([message, ("user", "Continue")])
    assert "custom_metric" not in json.dumps(requests[-1])
