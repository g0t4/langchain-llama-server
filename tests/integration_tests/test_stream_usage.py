from langchain_llama_server.chat_models import ChatLlamaServer
import rich

from tests.integration_tests.manual_tests.endpoints import LLAMA_SERVER_MUSE_BASE_URL, VLLM_INKLING2MINI_BASE_URL

class TestLlamaServer_Streaming_Manually:

    def test__when_stream_usage_is_true__finish_reason_and_stats_are_split_into_two_chunks():

        model = ChatLlamaServer(
            api_key="",
            model="",
            base_url=LLAMA_SERVER_MUSE_BASE_URL,
            debugme=True,
            # *** IMO don't use stream_usage=True with llama-server
            # *** instead use its `timings` object (on last SSE, or second to last if you also set stream_usage=True)
            stream_usage=True,
        )

        model.enable_tracking_input_chunks = True

        stream_chunks = model.stream(
            "what is your name?",
            max_tokens=200,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        )
        output_chunks = [chunk for chunk in stream_chunks]

        # * INPUT ASSERTIONS (ON llama-server's outputs)
        #  helps if llama-server's behavior changes
        #
        input_chunks = model._get_tracked_input_chunks()
        assert len(input_chunks) >= 2, "missing input chunks, should be 2+ input chunks"
        #
        # * second to last has finish_reason
        second_to_last_input = input_chunks[-2]
        assert len(second_to_last_input["choices"]) == 1
        assert second_to_last_input["choices"][0] is not None
        #
        # if not stop/length, then adjust test (if valid scenario)
        #
        # allow for model to ramble too much and get cut off with "length" seems to still behave the same as "stop"
        in_reason = second_to_last_input["choices"][0]["finish_reason"]
        assert in_reason in ("stop", "length")
        #
        # * last input chunk has usage -- stream_usage=True adds this extra trailing SSE with
        #   an empty choices list, which is also where llama-server moves `timings` to in this mode
        last_input = input_chunks[-1]
        assert last_input
        assert len(last_input["choices"]) == 0
        assert "usage" in last_input
        assert "timings" in last_input
        assert last_input["timings"]["prompt_n"] >= 1

        # * OUTPUT ASSERTIONS
        #
        # ChatLlamaServer self-reports the true last chunk via chunk_position="last" (see
        # _convert_chunk_to_generation_chunk), so langchain_core's BaseChatModel.stream() doesn't
        # need to append its own extra empty bonus chunk -- output_chunks lines up 1:1 with
        # input_chunks, no +1 to account for.
        assert len(output_chunks) == len(input_chunks)
        assert len(output_chunks) >= 2, "output chunks should be 2 or more"
        #
        # * second-to-last output chunk == second-to-last input chunk: has finish_reason, no stats yet
        second_to_last_output = output_chunks[-2]
        assert second_to_last_output is not None
        assert second_to_last_output.response_metadata is not None
        out_reason = second_to_last_output.response_metadata["finish_reason"]
        assert out_reason in ("stop", "length")
        assert not second_to_last_output.chunk_position, "finish_reason chunk isn't the true last one when stream_usage=True -- one more (usage/timings) chunk follows it"

        # * last output chunk == last input chunk: the usage/timings-only tail, self-reported as last
        last_output = output_chunks[-1]
        assert last_output is not None
        assert last_output.debug
        assert last_output.debug.timings
        assert last_output.usage_metadata
        assert last_output.chunk_position == "last"
        # FYI!! lesson is DO NOT ASSUME the position of debug (timings) / usage in terms of all chunks, just check and react when found


    def test__when_stream_usage_is_false__one_chunk_has_finish_reason_and_stats():

        model = ChatLlamaServer(
            api_key="",
            model="",
            base_url=LLAMA_SERVER_MUSE_BASE_URL,
            debugme=True,
            stream_usage=False,
        )

        model.enable_tracking_input_chunks = True

        stream_chunks = model.stream(
            "what is your name?",
            max_tokens=200,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        )
        output_chunks = [chunk for chunk in stream_chunks]

        # * INPUT ASSERTIONS (ON llama-server's outputs)
        #  helps if llama-server's behavior changes
        #
        input_chunks = model._get_tracked_input_chunks()
        assert len(input_chunks) >= 1, "missing input chunks, should be 1+ input chunks"
        #
        # * unlike stream_usage=True, there's no separate trailing usage-only SSE with an empty
        #   choices list -- the one and only last SSE carries BOTH finish_reason and timings
        last_input = input_chunks[-1]
        assert len(last_input["choices"]) == 1
        #
        # allow for model to ramble too much and get cut off with "length" seems to still behave the same as "stop"
        in_reason = last_input["choices"][0]["finish_reason"]
        assert in_reason in ("stop", "length")
        #
        assert last_input.get("usage") is None, "usage should be absent -- stream_usage=False was not requested"
        assert "timings" in last_input
        assert last_input["timings"]["prompt_n"] >= 1

        # * OUTPUT ASSERTIONS
        #
        # ChatLlamaServer self-reports the true last chunk via chunk_position="last" (see
        # _convert_chunk_to_generation_chunk), so langchain_core's BaseChatModel.stream() doesn't
        # need to append its own extra empty bonus chunk -- output_chunks lines up 1:1 with
        # input_chunks, no +1 to account for.
        assert len(output_chunks) == len(input_chunks)
        assert len(output_chunks) >= 1, "output chunks should be 1 or more"
        #
        # * last output chunk == last input chunk: carries finish_reason AND timings together,
        #   since they arrived on the same SSE this time, self-reported as last
        last_output = output_chunks[-1]
        assert last_output is not None
        assert last_output.response_metadata is not None
        out_reason = last_output.response_metadata["finish_reason"]
        assert out_reason in ("stop", "length")
        assert last_output.debug
        assert last_output.debug.timings
        assert last_output.usage_metadata is None, "usage_metadata should be absent -- stream_usage=False was not requested"
        assert last_output.chunk_position == "last"


def test__when_stream_usage_is_false__one_chunk_has_finish_reason_and_stats_vllm():
    print() # avoid print on test name line

    model = ChatLlamaServer(
        api_key="",
        model="",
        base_url=VLLM_INKLING2MINI_BASE_URL, # inkling-2-mini on vllm
        # base_url="http://build21.lan:8027/v1", # qwen3.8-27b on sglang
        debugme=True,
        stream_usage=False,
    )

    model.enable_tracking_input_chunks = True

    stream_chunks = model.stream(
        "what is 1+1? answer only",
        max_tokens=20,
    )
    output_chunks = [chunk for chunk in stream_chunks]

    # * INPUT ASSERTIONS (ON llama-server's outputs)
    #  helps if llama-server's behavior changes
    #
    input_chunks = model._get_tracked_input_chunks()
    rich.print("input_chunks", input_chunks)
    assert len(input_chunks) >= 1, "missing input chunks, should be 1+ input chunks"
    #
    # * unlike stream_usage=True, there's no separate trailing usage-only SSE with an empty
    #   choices list -- the one and only last SSE carries BOTH finish_reason and timings
    last_input = input_chunks[-1]
    # rich.print("last_input", last_input)
    assert len(last_input["choices"]) == 1
    #
    # allow for model to ramble too much and get cut off with "length" seems to still behave the same as "stop"
    in_reason = last_input["choices"][0]["finish_reason"]
    assert in_reason in ("stop", "length")
    #
    # * VLLM sends usage/metrics when stream_usage is included
    assert last_input.get("usage") is None, "`usage` should be absent -- `stream_usage=False` was not requested"
    assert last_input.get("metrics") is None, "`metrics` should be absent -- `stream_usage=False` was not requested"

    # # * OUTPUT ASSERTIONS
    # #
    # # ChatLlamaServer self-reports the true last chunk via chunk_position="last" (see
    # # _convert_chunk_to_generation_chunk), so langchain_core's BaseChatModel.stream() doesn't
    # # need to append its own extra empty bonus chunk -- output_chunks lines up 1:1 with
    # # input_chunks, no +1 to account for.
    # assert len(output_chunks) == len(input_chunks)
    # assert len(output_chunks) >= 1, "output chunks should be 1 or more"
    # #
    # # * last output chunk == last input chunk: carries finish_reason AND timings together,
    # #   since they arrived on the same SSE this time, self-reported as last
    # last_output = output_chunks[-1]
    # assert last_output is not None
    # assert last_output.response_metadata is not None
    # out_reason = last_output.response_metadata["finish_reason"]
    # assert out_reason in ("stop", "length")
    # assert last_output.debug
    # assert last_output.debug.timings
    # assert last_output.usage_metadata is None, "usage_metadata should be absent -- stream_usage=False was not requested"
    # assert last_output.chunk_position == "last"
    #
