from typing import Awaitable, Callable
import openai
from pydantic import Field, SecretStr
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai.chat_models.base import BaseChatOpenAI
from openai.types.chat import ChatCompletion

# NOTES:
#
# - Inspired by ChatXAI chat_model (which provides reasoning_content, citations, and other nuances of xAI):
#   - https://github.com/langchain-ai/langchain/blob/master/libs/partners/xai/langchain_xai/chat_models.py
#   - which I tried after I stumbled on xAI's docs which shows they carefully support OpenAI compatible endpoints:
#   - https://docs.x.ai/developers/rest-api-reference/inference/chat#chat-completions
#
# BaseChatOpenAI(
#   BaseChatModel(
#     BaseLanguageModel[AIMessage],ABC ))
#
# - BTW abc/ABC => constrains BaseChatModel to be an abstract class (not instantiable)
# - hover docs for BaseChatModel shows architeture of chat_models

def print_indented(obj, level: int = 1):
    import rich
    from rich.padding import Padding
    from rich.pretty import Pretty

    if isinstance(obj, str):
        what = obj
    else:
        what = Pretty(obj)
    rich.print(Padding(what, (0, 0, 0, 4 * level)))

class ChatLlamaServer(BaseChatOpenAI):

    # LEAVE as reminder I don't need this right now:
    # make model_name/model optional... actually it already is optional on BaseChatOpenAI:
    # model_name: str = Field(default="", alias="model")

    # * ONLY make pyright think `api_key=""` is allowed
    #   STOP IT WES
    #   DO NOT MAKE THIS DO ANYTHING ELSE, NOT WORTH IT, NO TIME
    # FYI this only allows you to set "" empty string an not have pyright complain
    #   w/o this pyright will put red squigglies under the api_key="" when using ChatLlamaServer(api_key="")
    #   PRN could rip out some/all of BaseChatOpenAI so I don't have to even set api_key which currently in validate_environment flips out if its empty
    openai_api_key: ( \
        SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]] | str
    ) = Field(default="", alias="api_key")

    # dump raw messages (i.e. raw SSE)
    troubleshootme: bool = Field(default=False, alias="debugme")

    # hide things like timings
    # probably best to summarize (summary() tool) what you print about messages
    #   quiet suppresses some extra fields on AIMessage[Chunk]s ... largely added to make the messages printable in a list :)
    quiet: bool = Field(default=False)

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        chat_result = super()._create_chat_result(response, generation_info)

        if self.troubleshootme:
            import rich
            rich.print("[bold gray0 on blue]SINGLE")
            print_indented("response")
            print_indented(response, level=2)
            print_indented("generation_info")
            print_indented(generation_info, level=2)  # None so far
            print_indented("chat_result")
            print_indented(chat_result, level=2)

        out_message = chat_result.generations[0].message
        out_message.response_metadata["model_provider"] = "llama_server"

        if type(response) is ChatCompletion:
            choice0 = response.choices[0]
            message = choice0.message
            if hasattr(message, "reasoning_content"):
                reasoning_content = getattr(message, "reasoning_content")
                out_message.additional_kwargs["reasoning_content"] = reasoning_content
        elif type(response) is dict:
            #  do I ever need dict type? if not remove it from type hints
            raise ValueError("TODO implement support for dict response type in ChatLlamaServer._create_chat_result")
        else:
            raise ValueError(f"Unexpected response format in ChatLlamaServer._create_chat_result: {type(response)}")

        # out_message is the message returned by invoke/stream/etc
        if hasattr(response, "timings") and not self.quiet:
            setattr(out_message, "timings", getattr(response, "timings"))
        if hasattr(response, "__verbose") and not self.quiet:
            # using verbose instead of __verbose b/c rich.print won't print __verbose... though maybe that is desirable?
            setattr(out_message, "verbose", getattr(response, "__verbose"))

        if self.troubleshootme:
            print_indented("out_message")
            print_indented(out_message, level=2)

        return chat_result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)

        if self.troubleshootme:
            import rich
            rich.print("[bold gray0 on blue]CHUNK")
            print_indented("raw_chunk")
            print_indented(chunk, level=2)
            print_indented("generation")
            print_indented(generation, level=2)

        if generation is None or generation.message is None:
            return None

        message = generation.message
        if message.response_metadata:
            message.response_metadata["model_provider"] = "llama_server"

        if chunk:
            delta = chunk["choices"][0]["delta"]
            if delta and "reasoning_content" in delta:
                message.additional_kwargs["reasoning_content"] = delta["reasoning_content"]

        # TODO if chunk is not a dict? like above for non-streaming?

        if "timings" in chunk and not self.quiet:
            message.timings = chunk["timings"]
        if "__verbose" in chunk and not self.quiet:
            message.verbose = chunk["__verbose"]

        # PRN ? hold over timings and __verbose for the last chunk too (or instead of the last SSE's chunk which is second to last chunk)? (has reasoning_content, content and full message)
        #   chunk_position="last"
        #   comes after all SSEs arrived

        return generation
