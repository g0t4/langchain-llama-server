"""Run one hosted case with LIVE_CHAT=1 pytest -q -s -k sglang_with_usage.

The HTTP hook buffers this short response so the assertions can compare actual
SSE data frames to the LangChain output from the very same request.
"""

import json
import os
from functools import reduce
from operator import add

import httpx
import pytest
from openai.types.chat import ChatCompletionChunk

from langchain_openai_compatible import AccumulatedCompletion, ChatOpenAICompatible, calculate_stats


pytestmark = pytest.mark.skipif(os.getenv("LIVE_CHAT") != "1", reason="set LIVE_CHAT=1")
BASE_URLS = {
    "llama_server": "http://build21.lan:8016/v1",
    "vllm": "http://build21.lan:8017/v1",
    "sglang": "http://build21.lan:8027/v1",
}
DIAGNOSTICS = ("usage", "timings", "metrics", "sglext")


def delta(event):
    return event["choices"][0]["delta"] if event["choices"] else {}


def finish_reason(event):
    return event["choices"][0].get("finish_reason") if event["choices"] else None


def diagnostics(event):
    return {key for key in DIAGNOSTICS if event.get(key) is not None}


def reasoning(fields):
    value = fields.get("reasoning_content")
    return fields.get("reasoning") if value is None else value


def has_text(event):
    fields = delta(event)
    return bool(fields.get("content") or reasoning(fields))


def stream(backend, usage, **extra_body):
    events = []

    def capture(response):
        response.read()
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data:"):
                data = line.removeprefix("data:").strip()
                if data != "[DONE]":
                    events.append(json.loads(data))

    prefix = backend.upper()
    url = os.getenv(f"{prefix}_BASE_URL", BASE_URLS[backend])
    with httpx.Client(event_hooks={"response": [capture]}, timeout=30) as http:
        client = ChatOpenAICompatible(
            base_url=url, model=os.getenv(f"{prefix}_MODEL", ""), api_key="",
            http_client=http, timeout=30, max_retries=0,
        )
        with client.stream_completion(
            "Count from 1 to 100.", max_tokens=16, stream_usage=usage,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, **extra_body},
        ) as stream_handle:
            chunks = list(stream_handle)
        saved = stream_handle.get_final_completion().model_dump_json()

    stats = calculate_stats(AccumulatedCompletion.model_validate_json(saved))
    if usage or backend == "llama_server":
        assert stats.input.tokens > 0
        assert stats.output.tokens > 0
        assert stats.output.tokens_per_second > 0
    else:
        assert stats.output.tokens is None
        assert stats.output.tokens_per_second is None

    print(f"\n{backend} {url} | model={events[0].get('model') if events else '?'} | stream_usage={usage}")
    print("SSE   delta fields             finish   diagnostics")
    for i, event in enumerate(events):
        fields = ",".join(k for k, v in delta(event).items() if v is not None)
        print(f"{i:>3}   {fields or '-':24} {finish_reason(event) or '-':8} {', '.join(sorted(diagnostics(event)))}")
    print("LangChain chunks (content, reasoning, finish, normalized usage, position)")
    for i, chunk in enumerate(chunks):
        print(i, repr(chunk.content), repr(chunk.additional_kwargs.get("reasoning_content")),
              chunk.response_metadata.get("finish_reason"), chunk.usage_metadata, chunk.chunk_position)
    first = chunks[0].response_metadata
    if backend == "llama_server":
        assert first["headers"]["server"] == "llama.cpp"
    else:
        assert first["headers"]["x-inference-backend"] == backend
    assert first["backend"] == reduce(add, chunks).response_metadata["backend"] == backend.replace("_", "-")
    return events, chunks


def assert_langchain(events, chunks):
    # Every server event has a corresponding message. LangChain then emits an empty end marker.
    *messages, end = chunks
    assert len(messages) == len(events)
    assert end.chunk_position == "last" and end.content == ""
    assert "raw_chunks" not in end.response_metadata

    for event, message in zip(events, messages):
        parsed = ChatCompletionChunk.model_validate(event).model_dump()
        assert message.response_metadata["raw_chunks"] == [parsed]
        assert message.chunk_position is None
        assert message.content == (delta(event).get("content") or "")
        assert message.additional_kwargs.get("reasoning_content") == reasoning(delta(event))
        assert message.response_metadata.get("finish_reason") == finish_reason(event)
        if usage := event.get("usage"):
            assert message.usage_metadata["input_tokens"] == usage["prompt_tokens"]
            assert message.usage_metadata["output_tokens"] == usage["completion_tokens"]
            assert message.usage_metadata["total_tokens"] == usage["total_tokens"]
        else:
            assert message.usage_metadata is None

    merged = reduce(add, chunks)
    assert merged.response_metadata["raw_chunks"] == [
        message.response_metadata["raw_chunks"][0] for message in messages
    ]


def test_llama_server_with_usage():
    events, chunks = stream("llama_server", True)
    # The final text delta precedes the finish. Timings move to the empty-choice usage tail.
    text, finish, stats = events[-3:]
    assert has_text(text) and not has_text(finish)
    assert finish_reason(finish) in ("stop", "length")
    assert stats["choices"] == []
    assert all(not diagnostics(e) for e in events[:-1])
    assert diagnostics(stats) == {"usage", "timings"}
    assert stats["timings"]["predicted_n"] > 0
    assert_langchain(events, chunks)


def test_llama_server_without_usage():
    events, chunks = stream("llama_server", False)
    # One finish event now carries timings even though usage is absent.
    text, finish = events[-2:]
    assert has_text(text) and not has_text(finish)
    assert finish_reason(finish) in ("stop", "length")
    assert all(not diagnostics(e) for e in events[:-1])
    assert diagnostics(finish) == {"timings"}
    assert finish["timings"]["predicted_n"] > 0
    assert_langchain(events, chunks)


def test_vllm_with_usage():
    events, chunks = stream("vllm", True)
    # vLLM can deliver the final text on the finish event itself.
    finish, stats = events[-2:]
    assert has_text(finish) and finish_reason(finish) == "length"
    assert all(not diagnostics(e) for e in events[:-1])
    assert stats["choices"] == []
    assert diagnostics(stats) == {"usage", "metrics"}
    assert "time_to_first_token_ms" in stats["metrics"]
    assert_langchain(events, chunks)


def test_vllm_without_usage():
    events, chunks = stream("vllm", False)
    # The text-bearing length cutoff ends the server stream. There is no metrics tail.
    finish = events[-1]
    assert has_text(finish) and finish_reason(finish) == "length"
    assert all(not diagnostics(e) for e in events)
    assert_langchain(events, chunks)


def test_sglang_with_usage():
    events, chunks = stream("sglang", True, return_spec_tokens_details=True,
                            return_cached_tokens_details=True, return_timings=True)
    # The deployed timing extension follows the separate speculative-decoding event.
    text, finish, spec, stats = events[-4:]
    assert has_text(text) and not has_text(finish)
    assert finish_reason(finish) in ("stop", "length")
    assert all(not diagnostics(e) for e in events[:-2])
    assert spec["choices"] == stats["choices"] == []
    assert diagnostics(spec) == {"sglext"}
    assert "spec_tokens_details" in spec["sglext"]
    assert diagnostics(stats) == {"usage", "timings"}
    assert isinstance(stats["timings"]["choices"], list)
    assert_langchain(events, chunks)


def test_sglang_without_usage():
    events, chunks = stream("sglang", False)
    # With the usage and extension flags off, a separate finish follows the final text.
    text, finish = events[-2:]
    assert has_text(text) and not has_text(finish)
    assert finish_reason(finish) in ("stop", "length")
    assert all(not diagnostics(e) for e in events)
    assert_langchain(events, chunks)
