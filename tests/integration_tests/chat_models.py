from __future__ import annotations
import os
import rich

# * autoreload changed modules (both `import` and `from` style imports)
in_nvim_notebook = os.getenv("NVIM")
if in_nvim_notebook:
    get_ipython().extension_manager.load_extension("autoreload")  # pyright: ignore
    get_ipython().run_line_magic('autoreload', 'complete --print')  # pyright: ignore

import rich
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langchain_llama_server.chat_models import ChatLlamaServer, print_indented

model = ChatLlamaServer(
    api_key="",
    model="",
    base_url="http://build21.lan:8012",
    # debugme=True,
)

#FYI there is some last ACTUAL chunk emitted from langchain that isn't the last in the SSE stream of chunks... has "last" on it... but prior ACTUAL has stop reason and timings so that's the last one I care about

# %% * streaming chat

stream_chunks = model.stream("what is your name?")
net = None
for chunk in stream_chunks:
    if net is None:
        net = chunk
    else:
        # very cool way to aggregate the chunks!
        net += chunk
    print_indented("ACTUAL:")
    print_indented(chunk, level=2)

print_indented(net, level=2)
assert "Qwen3.5-35B-A3B" in net.response_metadata["model_name"]

# %% * non-streaming chat

ai_message = model.invoke("what is your name?")
rich.print(ai_message)
assert ai_message.additional_kwargs["reasoning_content"] is not None
assert "Qwen3.5-35B-A3B" in ai_message.response_metadata["model_name"]

# %% * streaming non-reasoning

model_nonreasoning = ChatLlamaServer(
    api_key="",
    model="",
    base_url="http://build21.lan:8012",
    # debugme=True,
)
stream_chunks = model_nonreasoning.stream(
    "what is your name?",
    max_tokens=20,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    },
)
net = None
for chunk in stream_chunks:
    if net is None:
        net = chunk
    else:
        # very cool way to aggregate the chunks!
        net += chunk
    print_indented("ACTUAL:")
    print_indented(chunk, level=2)
print_indented(net, level=2)
assert "Qwen3.5" in net.response_metadata["model_name"]
assert "reasoning_content" not in net.additional_kwargs

# %% * non-streaming non-reasoning

net = model_nonreasoning.invoke(
    "what is your name?",
    max_tokens=20,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    },
)
rich.print(net)
assert "Qwen3.5" in net.response_metadata["model_name"]

# %% * agent w/ python tool function

from langchain.agents import create_agent

def calculator(expression: str):
    """ compute the value of an expression """
    # print(expression)
    return eval(expression)
    # raise RuntimeError("FAILS")

agent = create_agent(model, tools=[calculator])
messages = agent.invoke({"messages": [
    HumanMessage("Use the calculator to find the value of 12.42111*124.33434344"),
]})
rich.print(messages)
message_one = messages['messages'][1]
assert isinstance(message_one, AIMessage)
assert len(message_one.tool_calls) == 1
tool_call = message_one.tool_calls[0]
assert tool_call['name'] == "calculator"
assert isinstance(messages['messages'][2], ToolMessage)
