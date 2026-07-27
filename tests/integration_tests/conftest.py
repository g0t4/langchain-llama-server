import os

import pytest

from langchain_llama_server.chat_models import ChatLlamaServer

# override via env if you want to point these at a different running model
LLAMA_SERVER_HOST = os.environ.get("LLAMA_SERVER_TEST_HOST", "ask.lan")
LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_TEST_PORT", "8012"))
LLAMA_SERVER_BASE_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"


@pytest.fixture
def model() -> ChatLlamaServer:
    """A plain ChatLlamaServer against the test server.

    Tests using this fixture must not assume anything model-specific (exact
    model name/version, whether reasoning_content shows up, etc) -- whatever
    happens to be loaded on this port can change. The one legitimate
    exception in this suite is Qwen's enable_thinking chat-template toggle
    (see test_qwen_reasoning_toggle.py), which needs a Qwen model and says
    so in its name.
    """
    return ChatLlamaServer(api_key="", model="", base_url=LLAMA_SERVER_BASE_URL)


def stream_and_merge(chunks):
    """Accumulate a stream of AIMessageChunks into one merged message."""
    net = None
    for chunk in chunks:
        net = chunk if net is None else net + chunk
    return net
