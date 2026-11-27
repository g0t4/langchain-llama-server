## langchain-llama-server

This LangChain integration supports `llama-server` for remote inference over HTTP/S.

Normally you can use `ChatOpenAI` from [`langchain-openai`](https://pypi.org/project/langchain-openai/) for OpenAI compatible APIs. But, anything beyond the OpenAI spec is not supported. Hence why this integration exists.

This package was created to fill in the gaps between the OpenAI spec and extra fields llama-server returns:
- Access `reasoning_content` on both streaming and non-streaming chat completions. 
- Capture `timings`, `__verbose` and other llama-server specific response fields using ["verbose" request param](https://github.com/ggml-org/llama.cpp/pull/24864/).
- Preserve backend-supplied `metrics`, `timings`, and `sglext` in each message's
  `response_metadata`, for both JSON and streaming responses, including usage-only
  SSE events. These fields are retained even with `quiet=True` and are not sent
  back as conversation content. Backend-specific request flags are still the
  caller's responsibility.
- Uses BaseChatOpenAI from [`langchain-openai`](https://pypi.org/project/langchain-openai/) to implement a `chat_model` called `ChatLlamaServer`

Similar to other server integrations:
- [`ChatOllama`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/ollama/langchain_ollama/chat_models.py#L261)
- [`ChatXAI`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/xai/langchain_xai/chat_models.py#L43)
   - `ChatXAI` also works with `llama-server` for `reasoning_content` because they both implement the OpenAI spec closely
- [`ChatOpenRouter`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openrouter/langchain_openrouter/chat_models.py#L85) 

## TODO

- `LlamaServerLLM` ?
- `LlamaServerEmbedding` ?
- `/slots` endpoint -- introspect concurrent-request/context-slot usage
- `n_probs`/logprobs support

## What about ChatLlamaCpp?

The `langchain_community` package provides [`ChatLlamaCpp`](https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/chat_models/llamacpp.py#L61)? 
This chat model targets in-process hosting using llama.cpp. It doesn't support llama-server for remote hosting.


## ChatOpenAICompatible

Use this sibling client for SGLang, vLLM, or llama-server chat completions.
Pass backend-specific request options through `extra_body`.

```python
from langchain_openai_compatible import ChatOpenAICompatible

model = ChatOpenAICompatible(base_url="http://localhost:8000/v1", model="my-model", api_key="")
for chunk in model.stream("Hello", stream_usage=True):
    for event in chunk.response_metadata.get("raw_chunks", []):
        print(event)
```

Each backend message chunk carries a one-element `raw_chunks` list containing
the full parsed input to the LangChain converter, including SDK defaults and
backend-specific fields. Adding chunks together retains those events in order.
LangChain's synthetic final chunk has no backend event. For a non-streaming
`invoke()`, read the complete parsed response from
`message.response_metadata["raw_response"]`.

### Verify a hosted stream

The [live stream tests](langchain_openai_compatible/tests/live/test_stream_usage.py)
document the event order observed on the three configured hosts. Each case checks
the server's SSE payloads first, then checks LangChain's messages from the same
request. Run one case with output visible to compare the two sequences by index:

```sh
LIVE_CHAT=1 pytest -q -s langchain_openai_compatible/tests/live -k sglang_with_usage
```

Drop `-k sglang_with_usage` to run all six cases. To point a case at another
deployment, set `SGLANG_BASE_URL` and optionally `SGLANG_MODEL` (or use the
`VLLM_` or `LLAMA_SERVER_` prefix). The assertions describe these deployments,
including the custom SGLang timing extension. The short token budget exercises
vLLM's length cutoff.
