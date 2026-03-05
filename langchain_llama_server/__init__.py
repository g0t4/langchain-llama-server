"""This is the langchain_llama_server package.

Provides for langchain to interact with llama-server 
"""

from importlib import metadata
from importlib.metadata import PackageNotFoundError

from langchain_llama_server.chat_models import ChatLlamaServer
# TODO if I add these then re-export them too:
# from langchain_ollama.embeddings import OllamaEmbeddings
# from langchain_ollama.llms import OllamaLLM


def _raise_package_not_found_error() -> None:
    raise PackageNotFoundError


try:
    if __package__ is None:
        _raise_package_not_found_error()
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatLlamaServer",
    # "OllamaEmbeddings",
    # "OllamaLLM",
    "__version__",
]

