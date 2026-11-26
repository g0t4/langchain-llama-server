from typing import Awaitable, Callable, List
import openai
from pydantic import Field, SecretStr
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai.chat_models.base import BaseChatOpenAI
from openai.types.chat import ChatCompletion
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Optional, Any
from copy import deepcopy

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


@dataclass
class DebugInfo:
    timings: Optional[Any] = None
    verbose: Optional[Any] = None


def print_indented(obj, level: int = 1):
    import rich
    from rich.padding import Padding
    from rich.pretty import Pretty

    if isinstance(obj, str):
        what = obj
    else:
        what = Pretty(obj)
    rich.print(Padding(what, (0, 0, 0, 4 * level)))


def _preserve_response_diagnostics(message, response):
    # Response metadata stays out of subsequent inference input.
    for field in ("metrics", "timings", "sglext"):
        value = response.get(field) if isinstance(response, dict) else getattr(response, field, None)
        if value is not None:
            message.response_metadata[field] = deepcopy(value)


class ChatLlamaServer(BaseChatOpenAI):
    """ Connect to a remote llama-server instance
    debug info shows when either:
    1. llama-server --verbose
    2. per request set verbose=True on extra_body

    YOU MUST SET api_key="" (do not use None and do not leve it unset)
    """

    # LEAVE as reminder I don't need this right now:
    # make model_name/model optional... actually it already is optional on BaseChatOpenAI:
    # model_name: str = Field(default="", alias="model")

    # * ONLY make pyright think `api_key=""` is allowed
    #   STOP IT WES
    #   DO NOT MAKE THIS DO ANYTHING ELSE, NOT WORTH IT, NO TIME
    # FYI this only allows you to set "" empty string and not have pyright complain
    #   w/o this pyright will put red squigglies under the api_key="" when using ChatLlamaServer(api_key="")
    #   PRN could rip out some/all of BaseChatOpenAI so I don't have to even set api_key which currently in validate_environment flips out if its empty
    # TODO find a way to set default value of api_key to an empty string "" in here so consumers don't have to... IIRC there's an issue with BaseChatOpenAI that caused that to fall apart
    # TODO also change None => "" if consumers don't set it? or set it to None?
    # TODO update docstring above if I get api_key default fixed
    # TODO ask Qwen to look into this for me, Qwen had a half way decent suggestion that I tossed b/c it's just easier for the change I was then working on to pass api_key=""
    openai_api_key: ( \
        SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]] | str
    ) = Field(default="", alias="api_key")

    # dump raw messages (i.e. raw SSE)
    troubleshootme: bool = Field(default=False, alias="debugme")

    # hide things like timings
    # probably best to summarize (summary() tool) what you print about messages
    #   quiet suppresses some extra fields on AIMessage[Chunk]s ... largely added to make the messages printable in a list :)
    quiet: bool = Field(default=False)

    # track input chunks for testing only
    enable_tracking_input_chunks: bool = Field(default=False)
    tracked_input_chunks: List[dict] = Field(default_factory=list)

    def _get_tracked_input_chunks(self) -> List[dict]:
        return self.tracked_input_chunks.copy()

    def _clear_tracked_input_chunks(self) -> None:
        self.tracked_input_chunks.clear()

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        chat_result = super()._create_chat_result(response, generation_info)
        for generation in chat_result.generations:
            _preserve_response_diagnostics(generation.message, response)

        if self.enable_tracking_input_chunks:
            self.tracked_input_chunks.append(dict(response))

        if self.troubleshootme:
            import rich
            rich.print("[bold gray0 on blue]WHOLE")
            print_indented("[bold gray0 on light_steel_blue]INPUT RESPONSE")
            print_indented(response, level=2)
            print_indented("[bold gray0 on light_steel_blue]INPUT GENERATION_INFO")
            print_indented(generation_info, level=2)  # None so far
            print_indented("[bold gray0 on light_steel_blue]SUPER CHAT_RESULT")
            print_indented(chat_result, level=2)

        out_message = chat_result.generations[0].message
        out_message.response_metadata["model_provider"] = "llama_server"

        if type(response) is ChatCompletion:
            choice0 = response.choices[0]
            message = choice0.message
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content is None:
                reasoning_content = getattr(message, "reasoning", None)
            if reasoning_content is not None:
                out_message.additional_kwargs["reasoning_content"] = reasoning_content
        elif type(response) is dict:
            #  do I ever need dict type? if not remove it from type hints
            raise ValueError("TODO implement support for dict response type in ChatLlamaServer._create_chat_result")
        else:
            raise ValueError(f"Unexpected response format in ChatLlamaServer._create_chat_result: {type(response)}")

        # copy debug info using DebugInfo dataclass
        debug_info = DebugInfo(
            timings=getattr(response, "timings", None),
            verbose=getattr(response, "__verbose", None),
        )
        if not self.quiet and (debug_info.timings or debug_info.verbose):
            out_message.debug = debug_info

        # TODO set reponse_backend on results too?

        return chat_result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)

        if self.enable_tracking_input_chunks:
            self.tracked_input_chunks.append(chunk)

        if self.troubleshootme:
            import rich
            rich.print("[bold gray0 on #79bbf1] CHUNK ")
            purple = "#9a9bd5"
            print_indented(f"[bold gray0 on {purple}] INPUT CHUNK ")
            print_indented(chunk, level=2)
            print_indented(f"[bold gray0 on {purple}] INPUT default_chunk_class ")
            print_indented(default_chunk_class, level=2)
            print_indented(f"[bold gray0 on {purple}] INPUT base_generation_info ")
            print_indented(base_generation_info, level=2)
            print_indented(f"[bold gray0 on {purple}] SUPER GENERATION ")
            print_indented(generation, level=2)

        if generation is None or generation.message is None:
            return None

        message = generation.message
        _preserve_response_diagnostics(message, chunk)
        if message.response_metadata:
            message.response_metadata["model_provider"] = "llama_server"

        if chunk and chunk.get("choices"):
            # btw checking for "choices" matters when passing stream_usage=True in which case the last SSE has no choices at all
            #  and in this case the timings/__verbose are on second to last SSE
            #  TODO investigate if any other cases result in empty choices
            #  TODO investigate second to last SSE in usages of DebugInfo
            choice0 = chunk["choices"][0]
            delta = choice0["delta"]
            if delta:
                reasoning_content = delta.get("reasoning_content")
                if reasoning_content is None:
                    reasoning_content = delta.get("reasoning")
                if reasoning_content is not None:
                    message.additional_kwargs["reasoning_content"] = reasoning_content
            # 
            # * ok vllm sends finish_reason on last content chunk (SSE) so the usage/metrics arrive on a new SSE after the first one... thus finish_reason is not a good cue
            #    or I could lookup if stream_usage if False and then use finish_reason to avoid extra chunk but meh it's not the biggest deal to get one extra
            # finish_reason = choice0.get("finish_reason", None)
            # if finish_reason:
            #     # TODO how about expand this test case to all backends?
            #     #   TODO and update my tests to assert that there's never a stray (extra) output chunk vs input chunks b/c we always tell langchain when we hit the last chunk so it never generates a virtual one
            #     #   right now if I set stream_usage=True then I assert extra chunk
            #     system_fingerprint = chunk.get("system_fingerprint", "") or ""
            #     is_vllm = system_fingerprint.startswith("vllm")
            #     if is_vllm:
            #         import rich
            #         rich.print(f"[red]found finish reason {finish_reason=}[/]")
            #         # message.response_metadata["finish_reason"] = finish_reason
            #         message.chunk_position = "last"
            #         # TODO where to put this and for all backends?

        # copy debug info using DebugInfo dataclass (simplified)
        debug_info = DebugInfo(
            timings=chunk.get("timings"),
            verbose=chunk.get("__verbose"),
        )
        if not self.quiet and (debug_info.timings or debug_info.verbose):
            message.debug = debug_info
            # FYI consumers should do hasattr(message, "debug") to check since other providers won't set debug
            # thus not setting it to None in an else here kinda helps avoid other providers blowing up if consumers always try to use `mesesage.debug`

        # * self-report "last" chunk
        # self-report the true final chunk via chunk_position="last", instead of leaving it to
        # langchain_core's BaseChatModel.stream() -- which otherwise can't tell we're already
        # done, and appends its own extra empty AIMessageChunk after ours just to mark that.
        # llama-server always attaches `timings` to exactly one chunk: the true last one --
        # whether that's the finish_reason chunk (stream_usage=False) or a separate trailing
        # usage-only chunk with an empty choices list (stream_usage=True). Checking for
        # `timings` directly is simpler and more reliable than inferring "last" from
        # finish_reason/choices shape, which differs across those two modes.
        if debug_info.timings:
            message.chunk_position = "last"

        # TODO if vllm => usage/metrics (vllm) or finish_reason
        #    in vllm you wouldn't always have usage (i.e. if you didn't ask for stream_usage)
        #    so I am thinking finish_reason might work
        #    TODO can I detect vllm here? or just use finish_reason?
        #      system_fingerprint startswith("vllm") is the strongest signal
        # TODO if sglang? => usage/sglext/others?

        # TODO set reponse_backend on results too?

        return generation
