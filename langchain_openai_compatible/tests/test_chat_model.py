import json
from functools import reduce
from operator import add

import httpx
import pytest
from openai.types.chat import ChatCompletionChunk

from langchain_openai_compatible import ChatOpenAICompatible


def client_for(handler, **options):
    return ChatOpenAICompatible(
        base_url="http://backend.test/v1",
        model="test",
        api_key="",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **options,
    )


@pytest.mark.parametrize(("headers", "backend"), [
    ({"X-Inference-Backend": "vllm", "Server": "llama.cpp"}, "vllm"),
    ({"Server": "llama.cpp"}, "llama-server"),
    ({}, "sglang"),
])
def test_stream_retains_backend_events_after_finish_and_repeated_usage(headers, backend):
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
            200, headers={"Content-Type": "text/event-stream", **headers},
            text="".join(frames) + "data: [DONE]\n\n",
        )

    chunks = list(client_for(respond).stream("Hello", stream_usage=True))
    assert [chunk.chunk_position for chunk in chunks] == [None] * 4 + ["last"]
    assert "raw_chunks" not in chunks[-1].response_metadata
    captured = [chunk.response_metadata["raw_chunks"][0] for chunk in chunks[:-1]]
    assert captured == [ChatCompletionChunk.model_validate(event).model_dump() for event in events]
    merged = reduce(add, chunks)
    assert merged.response_metadata["raw_chunks"] == captured
    assert [c.response_metadata["model"] for c in chunks if "model" in c.response_metadata] == ["test"]
    assert merged.response_metadata["model"] == "test"
    assert merged.response_metadata["backend"] == backend
    if headers:
        assert chunks[0].response_metadata["backend"] == backend
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
        return httpx.Response(200, headers={"X-Inference-Backend": "sglang"}, json=payload)

    model = client_for(respond)
    message = model.invoke("Hello")
    assert message.response_metadata["model"] == payload["model"]
    assert message.response_metadata["backend"] == "sglang"
    assert message.response_metadata["headers"]["x-inference-backend"] == "sglang"
    raw = message.response_metadata["raw_response"]
    assert raw["choices"][0]["message"]["reasoning_content"] == "think"
    assert raw["custom_metric"] == payload["custom_metric"]
    assert message.response_metadata["model_provider"] == "openai_compatible"
    assert message.additional_kwargs["reasoning_content"] == "think"
    model.invoke([message, ("user", "Continue")])
    assert "custom_metric" not in json.dumps(requests[-1])

    payload["system_fingerprint"] = "vllm-test"
    metadata = client_for(respond, include_response_headers=False).invoke("Hello").response_metadata
    assert "headers" not in metadata
    assert metadata["backend"] == "vllm"


@pytest.mark.parametrize(("fields", "expected"), [
    ({"system_fingerprint": "vllm-test"}, "vllm"),
    ({"usage": {}, "metrics": {}}, "vllm"),
    ({"timings": {"choices": []}}, "sglang"),
    ({"sglext": {}}, "sglang"),
    ({"timings": {"prompt_n": 1}}, "llama-server"),
    ({"usage": {}}, None),
    ({"system_fingerprint": "fp_other"}, None),
])
def test_backend_inference(fields, expected):
    model = client_for(lambda request: None)
    result = model._create_chat_result({
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        **fields,
    })
    metadata = result.generations[0].message.response_metadata
    if expected is None:
        assert "backend" not in metadata
    else:
        assert metadata["backend"] == expected


def test_accumulated_completion_with_fragmented_tool_call_and_usage_snapshots():
    from langchain_openai_compatible import AccumulatedCompletion

    def tool_delta(**fields):
        return {"choices": [{"index": 0, "delta": fields}]}

    def stats(total):
        return {"choices": [], "usage": {
            "prompt_tokens": 2, "completion_tokens": total - 2, "total_tokens": total,
        }}

    events = [
        tool_delta(reasoning_content="need weather"),
        tool_delta(tool_calls=[{"index": 0, "id": "call1", "type": "function",
                                "function": {"name": "weather", "arguments": '{"city":'}}]),
        tool_delta(tool_calls=[{"index": 0, "function": {"arguments": '"Austin"}'}}]),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        stats(5), stats(6),
    ]

    def respond(request):
        frames = [f"data: {json.dumps(event)}\n\n" for event in events]
        return httpx.Response(200, headers={"X-Inference-Backend": "sglang",
                                            "Content-Type": "text/event-stream"},
                              text="".join(frames) + "data: [DONE]\n\n")

    model = client_for(respond)
    with model.stream_completion("Get the weather") as stream:
        chunks = list(stream)
    result = stream.get_final_completion()
    assert result.message.tool_calls[0]["args"] == {"city": "Austin"}
    assert result.message.additional_kwargs["reasoning_content"] == "need weather"
    assert result.message.response_metadata["backend"] == "sglang"
    assert result.message.response_metadata["finish_reason"] == "tool_calls"
    assert result.message.usage_metadata["total_tokens"] == 6
    assert result.raw_chunks == [raw for c in chunks for raw in c.response_metadata.get("raw_chunks", [])]
    assert "raw_chunks" not in result.message.response_metadata
    assert AccumulatedCompletion.model_validate_json(result.model_dump_json()) == result

    with model.stream_completion("Get the weather") as partial:
        next(partial)
    assert len(partial.raw_chunks) == 1
    with pytest.raises(RuntimeError, match="exhausted"):
        partial.get_final_completion()


def test_async_stream_error_retains_partial_trace():
    import asyncio

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
            raise httpx.ReadError("lost stream")

    async def run():
        def respond(request):
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                                  stream=BrokenStream())

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            model = ChatOpenAICompatible(base_url="http://backend.test/v1", model="test",
                                         api_key="", http_async_client=http)
            with pytest.raises(httpx.ReadError, match="lost stream"):
                async with model.astream_completion("Hello") as stream:
                    async for chunk in stream:
                        assert chunk.content == "hi"
            assert stream.get_trace().outcome == "error"
            assert stream.get_trace().error_type == "ReadError"
            assert len(stream.raw_chunks) == 1
            with pytest.raises(RuntimeError, match="exhausted"):
                stream.get_final_completion()

    asyncio.run(run())


def test_client_timing_starts_on_first_pull_and_preserves_event_offsets(monkeypatch):
    from langchain_core.messages import AIMessageChunk
    from langchain_openai_compatible import AccumulatedCompletion
    from langchain_openai_compatible.completion import CompletionStream

    ticks = iter([10, 12, 15, 18, 20, 25])
    monkeypatch.setattr("langchain_openai_compatible.completion.monotonic_ns",
                        lambda: next(ticks) * 1_000_000)
    event = {"raw_chunks": [{"choices": []}]}
    stream = CompletionStream(iter([
        AIMessageChunk(content="", response_metadata=event),
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "think"}, response_metadata=event),
        AIMessageChunk(content="yes", response_metadata=event),
        AIMessageChunk(content="", chunk_position="last"),
    ]))
    assert stream.client_timing is None
    next(stream)
    assert stream.client_timing.first_event_ms == 2
    assert stream.client_timing.first_answer_ms is None
    list(stream)
    completion = stream.get_final_completion()
    timing = completion.client_timing
    assert timing.event_offsets_ms == [2, 5, 8]
    assert timing.first_reasoning_ms == 5
    assert timing.first_answer_ms == 8
    assert timing.elapsed_ms == 15
    assert timing.started_at.utcoffset().total_seconds() == timing.ended_at.utcoffset().total_seconds() == 0
    assert AccumulatedCompletion.model_validate_json(completion.model_dump_json()) == completion
    assert AccumulatedCompletion.model_validate(completion.model_dump(exclude={"client_timing"})).client_timing is None


@pytest.mark.parametrize("cause,outcome", [(None, "cancelled"), (KeyboardInterrupt, "cancelled"),
                                          (ValueError, "error")])
def test_serializable_interruption(cause, outcome):
    from langchain_core.messages import AIMessageChunk
    from langchain_openai_compatible import InterruptedCompletion, calculate_stats, capture_completion

    def chunks():
        yield AIMessageChunk(content="hi", response_metadata={
            "raw_chunks": [{"usage": {"completion_tokens": 2}}],
            "backend": "vllm", "model": "served-id", "headers": {"server": "uvicorn"},
        })
        if cause:
            raise cause("stopped")

    with capture_completion(chunks()) as stream:
        next(stream)
        if cause:
            with pytest.raises(cause):
                next(stream)
    partial = InterruptedCompletion.model_validate_json(stream.get_trace().model_dump_json())
    assert partial.outcome == outcome
    assert partial.model == "served-id"
    assert partial.backend == "vllm" and partial.headers["server"] == "uvicorn"
    assert partial.error_type == (cause.__name__ if cause else None)
    assert partial.client_timing.ended_at and partial.client_timing.elapsed_ms >= 0
    assert calculate_stats(partial).output.tokens == 2
    with pytest.raises(RuntimeError):
        stream.get_final_completion()


def test_interruption_before_first_event():
    from langchain_openai_compatible import capture_completion

    def failing_request():
        raise ConnectionError("connection refused")
        yield

    with capture_completion(failing_request()) as unused:
        pass
    assert unused.get_trace().outcome == "cancelled"
    assert unused.get_trace().client_timing is None

    with pytest.raises(ConnectionError):
        with capture_completion(failing_request()) as failed:
            next(failed)
    trace = failed.get_trace()
    assert trace.outcome == "error" and trace.raw_chunks == []
    assert trace.error_message == "connection refused"
    assert trace.client_timing.ended_at is not None
    assert trace.client_timing.first_event_ms is None
