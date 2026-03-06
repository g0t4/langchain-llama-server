from __future__ import annotations
import os
import rich

# * autoreload changed modules (both `import` and `from` style imports)
in_nvim_notebook = os.getenv("NVIM")
if in_nvim_notebook:
    get_ipython().extension_manager.load_extension("autoreload")  # pyright: ignore
    get_ipython().run_line_magic('autoreload', 'complete --print')  # pyright: ignore

import rich
from langchain_core.messages import HumanMessage
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

ai_message = model.invoke("what is your name?", store=True)
rich.print(ai_message)
assert ai_message.additional_kwargs["reasoning_content"] is not None
assert "Qwen3.5-35B-A3B" in ai_message.response_metadata["model_name"]


