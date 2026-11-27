"""The opt-in `verbose: true` extra_body flag -- returns llama-server's full
generation-settings dump (`__verbose`), in addition to `timings` (which is
always present regardless of this flag -- see test_stream_usage.py)."""


def test_non_streaming_verbose_extra_body(model):
    ai_message = model.invoke(
        "what is your name?",
        max_tokens=1,
        extra_body={"verbose": True},
    )
    assert ai_message.debug is not None
    assert ai_message.debug.verbose is not None
    assert ai_message.debug.timings is not None


def test_streaming_verbose_extra_body(model):
    chunks = list(model.stream(
        "what is your name?",
        max_tokens=1,
        extra_body={"verbose": True},
    ))

    # the chunk carrying timings/__verbose self-reports chunk_position="last"
    # (see ChatLlamaServer._convert_chunk_to_generation_chunk), so it's always
    # the real last chunk now -- no need to scan for it.
    last = chunks[-1]
    assert last.chunk_position == "last"
    assert last.debug is not None
    assert last.debug.verbose is not None
    assert last.debug.timings is not None
