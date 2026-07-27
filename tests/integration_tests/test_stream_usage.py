from langchain_llama_server.chat_models import ChatLlamaServer, print_indented


def test_last_sse_has_empty_choices():

    model = ChatLlamaServer(
        api_key="",
        model="",
        base_url="http://ask.lan:8012",
        debugme=True,
        # *** IMO don't use stream_usage=True with llama-server
        # *** instead use its `timings` object (on last SSE, or second to last if you also set stream_usage=True)
        stream_usage=True)
    stream_chunks = model.stream(
        "what is your name?",
        max_tokens=200,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    )
    net = None
    all = []
    for chunk in stream_chunks:
        all.append(chunk)
        if net is None:
            net = chunk
        else:
            # very cool way to aggregate the chunks!
            net += chunk
        print_indented("ACTUAL:")
        print_indented(chunk, level=2)
    print_indented(net, level=2)

    # assert net.choices[-1].delta.content == ""

    # FYI round about way to test this as the mapping to AIMessageChunk cannot validate choices was empty
    #  so lets look for some evidence that this happened which means timings was on second to last SSE
    print("second to last")
    second_to_last = all[-2]
    assert second_to_last is not None
    assert second_to_last.debug
    assert second_to_last.debug.timings
    print_indented(second_to_last)
    print()
    print("last")
    last = all[-1]
    assert last is not None
    assert "debug" not in last
    print_indented(last)


test_last_sse_has_empty_choices()
