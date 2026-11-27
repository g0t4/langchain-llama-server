"""Qwen-specific: the enable_thinking chat-template toggle.

This is the one deliberately model-specific file in this suite -- it needs
a Qwen model on the test port, since enable_thinking is a Qwen chat-template
feature, not something assumed to exist elsewhere. Every other test in this
suite should stay model-agnostic (see conftest.py).
"""

import pytest

from langchain_llama_server.chat_models import ChatLlamaServer

from .conftest import LLAMA_SERVER_BASE_URL, stream_and_merge


@pytest.fixture
def qwen_no_thinking() -> ChatLlamaServer:
    return ChatLlamaServer(
        api_key="",
        model="",
        base_url=LLAMA_SERVER_BASE_URL,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def test_reasoning_present_by_default(model):
    ai_message = model.invoke("what is your name?", max_tokens=50)
    assert "Qwen" in ai_message.response_metadata.get("model_name", ""), \
        "this test needs a Qwen model on the test port (see conftest.py LLAMA_SERVER_* env vars)"
    assert ai_message.additional_kwargs.get("reasoning_content")


def test_enable_thinking_false_disables_reasoning_non_streaming(qwen_no_thinking):
    ai_message = qwen_no_thinking.invoke("what is your name?", max_tokens=20)
    assert "reasoning_content" not in ai_message.additional_kwargs


def test_enable_thinking_false_disables_reasoning_streaming(qwen_no_thinking):
    net = stream_and_merge(qwen_no_thinking.stream("what is your name?", max_tokens=20))
    assert "reasoning_content" not in net.additional_kwargs
