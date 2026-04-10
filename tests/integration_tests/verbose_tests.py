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
    base_url="http://ask.lan:8012",
    debugme=True,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False  # for qwen3.5 testing
        }
    },
)

# %% * non-streaming chat copies __verbose

ai_message = model.invoke(
    "what is your name?",
    max_tokens=1,
    extra_body={
        # setting verbose ensures __verbose is returned
        "verbose": True,
    },
)
# must have --verbose on llama-server to get this to work, or pass verbose in extra_body (above)
rich.print(ai_message)
assert ai_message is not None
debug = ai_message.debug
assert debug is not None
assert debug.verbose is not None
assert debug.timings is not None

# %% * streaming sets __verbose

last_sses_chunk = None
for chunk in model.stream(
        "what is your name?",
        max_tokens=1,
        extra_body={
            # setting verbose ensures __verbose is returned
            "verbose": True,
        },
):
    if "finish_reason" in chunk.response_metadata is not None:
        last_sses_chunk = chunk

rich.print(last_sses_chunk)
assert last_sses_chunk is not None
debug = last_sses_chunk.debug
assert debug is not None
assert debug.verbose is not None
assert debug.timings is not None
# FYI just leave as is on last SSE's chunk only is fine for now
#   this is not on the last chunk though which comes after last SSE
#   that is fine, I can capture it from second to last SSE's chunk
#   PRN I could also extend the chunk type to copy over verbose/timings when I do (net += chunk) in add operation overload
