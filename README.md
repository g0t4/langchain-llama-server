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

This combines what I learned building an dusing the first ChatLlamaServer.
I really find myself wanting the original input chunks attached to the langchain output chunks.
So that's what codex and I setup here!

Goals:
- Do the heavy lifting in ONE spot for all my python clients!
  - leverage best of both worlds:
    - original server chunks (SSEs)
    - langchain projections, i.e. `AIMessageChunk`
- Project standardized reasoning onto `AIMessageChunk`
- Preserve metrics
  - `llama-server`'s `timings` and `usage`
  - `sglang`'s `usage`, `timings.choices` + my own `sglext` (sglang extension)
  - `vllm`'s `usage` and `metrics`
  - AND other backends as needed
  - NOTE all three provide openai compatible `usage`
  - no more coupling to a limited list of metric (and other) fields
- Explicitly work with llama-server, vllm, sglang
  - as needed, expand to any openai compat backend
- Store the server-reported model ID in `response_metadata["model"]` on JSON completions and the first stream event carrying a nonempty model. Interrupted traces expose it as `trace.model` when observed. The configured request alias may differ.
- ☑ Infer `backend` and store it in `response_metadata["backend"]`.
  - The deployment can declare its name in the `X-Inference-Backend` HTTP response header. This takes precedence over `Server: llama.cpp` and payload inference. Headers are captured by default in `response_metadata["headers"]` on the first stream chunk or completed response. Set `include_response_headers=False` for payload-only inference.
  - `vllm` from its `vllm-` fingerprint or the combination of `usage` and `metrics`.
  - `sglang` from `sglext`, `timings.choices`, or a choice carrying `matched_stop`.
  - `llama-server` from its build fingerprint (e.g. `b12230-a4d4f2fd4`) or a `timings` object without `choices`.
  - The first identifiable stream event carries the label, so adding the chunks retains it once. Unrecognized responses omit the key.
- A client often needs `SSE`s for making an interactive, progress focused experience for consumers:
  - Users need to see that a model is working
  - Users need to see what the model is doing
  - Users need to be able to stop work that is wrong for any number of reasons:
    - Model misunderstood
    - Stupid tool calling decisions
    - Loops (reasoning, tool use, etc)
    - Steering/providing more context
- A client ALSO often needs the full completion object once streaming is done
  - Why make clients combine SSEs into a chunk, we can do that HERE!
  - ☑ Return an `AccumulatedCompletion` after streaming. Keep the backend events in
    order and assemble the assistant message with the last reported cumulative usage.
- ☑ Calculate standardized stats on demand from a completion or saved trace.
- DOCUMENT:
  - with live tests against backends
  - what each backend returns in each scenario
  - what comes out of our langchain client in each scenario too
  - how we combine input chunk preservation across SSEs

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

### Save an accumulated completion

```python
with model.stream_completion("Hello", stream_usage=True) as stream:
    for chunk in stream:
        print(chunk.text, end="")
    completion = stream.get_final_completion()

trace_json = completion.model_dump_json()
```

`completion.message` is the assembled LangChain `AIMessage`, including reasoning,
tool calls, headers, and backend identity when supplied. `completion.raw_chunks`
holds the ordered parsed backend events once in the serialized trace.
`AccumulatedCompletion.model_validate_json(trace_json)` reads it back.

For async clients, use `async with model.astream_completion(...)` and `async for`.
The final result becomes available when iteration runs to exhaustion. If the
consumer breaks early or the request fails, `stream.raw_chunks` contains the
events captured so far and `get_final_completion()` raises `RuntimeError`.

### Calculate standardized stats

```python
from langchain_openai_compatible import AccumulatedCompletion, calculate_stats

stats = calculate_stats(completion)  # also accepts the AIMessage returned by invoke()
print(stats.input.cached_tokens, stats.output.tokens_per_second)
replayed = calculate_stats(AccumulatedCompletion.model_validate_json(trace_json))
```

The calculator reads the last non-null report of each kind independently, so
usage snapshots count once and diagnostics on separate events survive. Values of
`None` mean unavailable. Reported zeros survive. Calculations run on demand,
including after loading an older trace.

| Field | Meaning |
|---|---|
| `input.tokens`, `cached_tokens` | Total prompt tokens and the subset served from cache. llama-server can supply both through timings alone. |
| `input.tokens_per_second` | Prompt processing throughput. llama-server reports it directly. SGLang uses uncached tokens divided by the forward-entry to prefill-finished interval. |
| `output.tokens_per_second` | Decode throughput. llama-server and SGLang report it directly. For vLLM use 1000 divided by mean inter-token latency in milliseconds. |
| `output.tokens`, `reasoning_tokens`, `non_reasoning_tokens` | Output total, reported reasoning count, and their difference when valid. Non-reasoning output can include tool calls. |
| `output.draft` | Drafted and accepted token counts, acceptance rate (fraction), and reported mean acceptance length. A reported rate takes precedence over a ratio derived from counts. |
| `time_to_first_token_ms`, `queue_ms` | Server-reported TTFT and queue time, converted to milliseconds. |

With multiple choices the calculator exposes request-wide usage and omits
choice-specific performance statistics. A plain OpenAI-compatible usage report
also works without a recognized backend.

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

### Client timing

`completion.client_timing` records UTC `started_at` and `ended_at` timestamps
and monotonic `elapsed_ms` from the first pull through natural stream exhaustion.
`event_offsets_ms[i]` is when the client observed `raw_chunks[i]`.
`first_event_ms`, `first_reasoning_ms`, and `first_answer_ms` distinguish the
first backend event from the first nonempty reasoning and answer text. Missing
milestones remain null. Older traces load with `client_timing=None`.

The same timing object is available as `stream.client_timing` during iteration.
These observations include SDK processing and time the consumer spends between
pulls. Server-reported TTFT remains available through `calculate_stats()`.

### Save a stopped or failed stream

After the stream context exits, `stream.get_trace()` returns either the completed
`AccumulatedCompletion` or an `InterruptedCompletion` with outcome `cancelled`
or `error`. Both serialize with `model_dump_json()`. An early break, Ctrl+C, or
async task cancellation closes the stream. A failed request preserves its exception
type and message and propagates the original exception. Started requests retain
their end time and elapsed time even if no backend event arrived.

```python
try:
    with model.stream_completion(messages) as stream:
        for chunk in stream:
            render(chunk)
finally:
    trace_file.write(stream.get_trace().model_dump_json() + "\n")
```

For a runnable returned by `bind_tools()`, use
`capture_completion(bound.stream(messages))` or
`acapture_completion(bound.astream(messages))` to preserve the binding's request
options. `calculate_stats()` also accepts interrupted traces and reports whatever
measurements arrived before the interruption.
