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
    extra_body={"chat_template_kwargs": {
        "enable_thinking": False
    }},
)

# %% * non-streaming chat copies __verbose

ai_message = model.invoke(
    "what is your name?",
    max_tokens=1,
    store=True,
)
rich.print(ai_message)
assert hasattr(ai_message, "verbose")  # must have --verbose on llama-server to get this to work

# %% * streaming sets __verbose
