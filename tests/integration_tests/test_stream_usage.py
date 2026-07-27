from langchain_core.messages import AIMessageChunk
from langchain_llama_server.chat_models import ChatLlamaServer, print_indented


def test__second_to_last_has_finish_reason__last_chunk_has_timing():

    model = ChatLlamaServer(
        api_key="",
        model="",
        base_url="http://ask.lan:8012",
        debugme=True,
        # *** IMO don't use stream_usage=True with llama-server
        # *** instead use its `timings` object (on last SSE, or second to last if you also set stream_usage=True)
        stream_usage=True,
        # !TODO! add test of stream_usage=False (finish_reason + timings both on last SSE)
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
    # net = sum(all, AIMessageChunk(content=''))
    # print("NET", net)

    # * INPUT ASSERTIONS (ON llama-server's outputs)
    #  helps if llama-server's behavior changes
    #
    input_chunks = model._get_tracked_input_chunks()
    # print("INPUT_CHUNKS:")
    # print_indented(input_chunks, 2)
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
    # * last input chunk has usage
    last_input = input_chunks[-1]
    assert last_input
    assert len(last_input["choices"]) == 0
    assert "usage" in last_input
    assert "timings" in last_input
    assert last_input["timings"]["prompt_n"] >= 1

    # * OUTPUT ASSERTIONS
    assert len(output_chunks) >= 2, "output chunks should be 3 or more"
    #
    third_to_last_output = output_chunks[-3]
    assert third_to_last_output is not None
    assert third_to_last_output.response_metadata is not None
    out_reason = third_to_last_output.response_metadata["finish_reason"]
    assert out_reason in ("stop", "length")

    second_to_last_output = output_chunks[-2]
    assert second_to_last_output is not None
    assert second_to_last_output.debug
    assert second_to_last_output.debug.timings
    assert second_to_last_output.usage_metadata
    # FYI!! lesson is DO NOT ASSUME the position of debug (timings) / usage in terms of all chunks, just check and react when found

    # ? verify in openai langchain provider if this is true:
    # * IIAC the last output chunk corresponds to [DONE] SSE?
    last_output = output_chunks[-1]
    assert last_output is not None


test__second_to_last_has_finish_reason__last_chunk_has_timing()
