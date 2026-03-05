## langchain-llama-server

This LangChain integration supports `llama-server` for remote inference over HTTP/S.

Normally you can use `ChatOpenAI` from [`langchain-openai`](https://pypi.org/project/langchain-openai/) for OpenAI compatible APIs. But, anything beyond the OpenAI spec is not supported. Hence why this integration exists.

This package was created to fill in the gaps between the OpenAI spec and extra fields llama-server returns:
- Access `reasoning_content` on both streaming and non-streaming chat completions. 
- Capture `timings`, `__verbose` and other llama-server specific response fields.
- Uses BaseChatOpenAI from [`langchain-openai`](https://pypi.org/project/langchain-openai/) to implement a `chat_model` called `ChatLlamaServer`

Similar to other server integrations:
- [`ChatOllama`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/ollama/langchain_ollama/chat_models.py#L261)
- [`ChatXAI`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/xai/langchain_xai/chat_models.py#L43)
   - `ChatXAI` also works with `llama-server` for `reasoning_content` because they both implement the OpenAI spec closely
- [`ChatOpenRouter`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openrouter/langchain_openrouter/chat_models.py#L85) 

## TODO

- `LlamaServerLLM` ?
- `LlamaServerEmbedding` ?

## What about ChatLlamaCpp?

The `langchain_community` package provides [`ChatLlamaCpp`](https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/chat_models/llamacpp.py#L61)? 
This chat model targets in-process hosting using llama.cpp. It doesn't support llama-server for remote hosting.

