from langchain_llama_server.chat_models import ChatLlamaServer
import rich

from tests.integration_tests.manual_tests.endpoints import LLAMA_SERVER_MUSE_BASE_URL, SGLANG_QWEN38_BASE_URL, VLLM_INKLING2MINI_BASE_URL


class TestLlamaServer_Streaming_Manually:

    def test__when_stream_usage_is_true__finish_reason_and_stats_are_split_into_two_chunks(self):
        print()  # avoid print on test name line

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
            "what is 1+1? answer only",
            max_tokens=200,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        )
        output_chunks = [chunk for chunk in stream_chunks]

        # * INPUT ASSERTIONS (ON llama-server's outputs)
        #  these are merely to troubleshoot server responses (aka inputs into langchain)
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
        assert "finish_reason" in second_to_last_input["choices"][0]
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
        output_finish_reason = second_to_last_output.response_metadata["finish_reason"]
        assert output_finish_reason in ("stop", "length")
        assert not second_to_last_output.chunk_position, "finish_reason chunk isn't the true last one when stream_usage=True -- one more (usage/timings) chunk follows it"

        # * last output chunk == last input chunk: the usage/timings-only tail, self-reported as last
        last_output = output_chunks[-1]
        assert last_output is not None
        assert last_output.debug
        assert last_output.debug.timings
        assert last_output.usage_metadata
        assert last_output.chunk_position == "last"
        # FYI!! lesson is DO NOT ASSUME the position of debug (timings) / usage in terms of all chunks, just check and react when found

    def test__when_stream_usage_is_false__one_chunk_has_finish_reason_and_stats(self):
        print()  # avoid print on test name line

        model = ChatLlamaServer(
            api_key="",
            model="",
            base_url=LLAMA_SERVER_MUSE_BASE_URL,
            debugme=True,
            stream_usage=False,
        )

        model.enable_tracking_input_chunks = True

        stream_chunks = model.stream(
            "what is 1+1? answer only",
            max_tokens=200,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        )
        output_chunks = [chunk for chunk in stream_chunks]

        # * INPUT ASSERTIONS (ON llama-server's outputs)
        #  these are merely to troubleshoot server responses (aka inputs into langchain)
        #
        input_chunks = model._get_tracked_input_chunks()
        assert len(input_chunks) >= 1, "missing input chunks, should be 1+ input chunks"
        #
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
        assert "finish_reason" in last_output.response_metadata
        output_finish_reason = last_output.response_metadata["finish_reason"]
        assert output_finish_reason in ("stop", "length")
        assert last_output.debug
        assert last_output.debug.timings
        assert last_output.usage_metadata is None, "usage_metadata should be absent -- stream_usage=False was not requested"
        assert last_output.chunk_position == "last"


class TestVLLM_Streaming_Manually:

    # FYI run just one:     ptw -- -k "TestVLLM_Streaming_Manually and true" --capture no

    def test__when_stream_usage_is_true__finish_reason_and_stats_are_split_into_two_separate_chunks(self):

        print()  # avoid print on test name line
        model = ChatLlamaServer(
            api_key="",
            model="",
            base_url=VLLM_INKLING2MINI_BASE_URL,
            debugme=True,
            stream_usage=True,
        )

        model.enable_tracking_input_chunks = True

        stream_chunks = model.stream(
            "what is 1+1? answer only",
            max_tokens=200,
        )
        output_chunks = [chunk for chunk in stream_chunks]

        # * INPUT ASSERTIONS (ON VLLM's outputs)
        #  these are merely to troubleshoot server responses (aka inputs into langchain)
        #
        input_chunks = model._get_tracked_input_chunks()
        assert len(input_chunks) >= 1, "missing input chunks, should be 1+ input chunks"
        #
        last_input = input_chunks[-1]
        rich.print(f"{last_input=}")
        assert len(last_input["choices"]) == 0
        #
        second_to_last_input = input_chunks[-2]
        rich.print(f"{second_to_last_input=}")
        assert len(second_to_last_input["choices"]) == 1
        #
        # allow for model to ramble too much and get cut off with "length" seems to still behave the same as "stop"
        in_reason = second_to_last_input["choices"][0]["finish_reason"]
        assert in_reason in ("stop", "length")
        #
        assert second_to_last_input.get("usage") is None, "`usage` should be absent -- `stream_usage=False` was not requested"
        assert second_to_last_input.get("metrics") is None, "`metrics` should be absent -- `stream_usage=False` was not requested"
        #
        # VLLM notes:
        # - often multiple tokens per SSE (good idea IMO)

        # * OUTPUT ASSERTIONS
        #
        rich.print(output_chunks)
        assert len(output_chunks) >= 1, "output chunks should be 1 or more"
        assert len(output_chunks) == len(input_chunks)

        # * second to last has finish_reason
        #   as well as the last content/reasoning or both (not testing that here)
        chunk_with_finish_reason = output_chunks[-2]
        assert chunk_with_finish_reason is not None
        assert chunk_with_finish_reason.response_metadata is not None
        assert "finish_reason" in chunk_with_finish_reason.response_metadata
        output_finish_reason = chunk_with_finish_reason.response_metadata["finish_reason"]
        assert output_finish_reason in ("stop", "length")

        # * last chunk has metrics
        chunk_with_metrics = output_chunks[-1]
        assert "metrics" in chunk_with_metrics.response_metadata  # copied by _preserve_response_disagnostics
        assert "usage" in chunk_with_metrics.response_metadata  # copied by _preserve_response_disagnostics

        assert chunk_with_metrics.chunk_position == "last"

    def test__when_stream_usage_is_false(self):
        # ptw -- -k "TestVLLM_Streaming_Manually and false" --capture no
        #  DO NOT run multiple tests at a time (the output can overlap)

        print()  # avoid print on test name line

        model = ChatLlamaServer(
            api_key="",
            model="",
            base_url=VLLM_INKLING2MINI_BASE_URL,
            debugme=True,
            stream_usage=False,
        )

        model.enable_tracking_input_chunks = True

        stream_chunks = model.stream(
            "what is 1+1? answer only",
            max_tokens=200,
        )
        output_chunks = [chunk for chunk in stream_chunks]

        # * INPUT ASSERTIONS (ON VLLM's outputs)
        #  these are merely to troubleshoot server responses (aka inputs into langchain)
        #
        input_chunks = model._get_tracked_input_chunks()
        assert len(input_chunks) >= 1, "missing input chunks, should be 1+ input chunks"
        #
        last_input = input_chunks[-1]
        assert len(last_input["choices"]) == 1
        #
        # allow for model to ramble too much and get cut off with "length" seems to still behave the same as "stop"
        in_reason = last_input["choices"][0]["finish_reason"]
        assert in_reason in ("stop", "length")
        #
        assert last_input.get("usage") is None, "`usage` should be absent -- `stream_usage=False` was not requested"
        assert last_input.get("metrics") is None, "`metrics` should be absent -- `stream_usage=False` was not requested"
        #
        # VLLM notes:
        # - often multiple tokens per SSE (good idea IMO)

        # * OUTPUT ASSERTIONS
        #
        rich.print(output_chunks)
        assert len(output_chunks) >= 1, "output chunks should be 1 or more"
        assert len(output_chunks) == len(input_chunks) + 1
        # TODO do I care about marking last chunk so langchain doesn't add that for me as an extra chunk after last?
        #   in llama-server I use timings to do so... I could use metrics/usage to do so for vllm
        #   but that requires asking for usage or metrics (which I'd always be doing in my scenarios)
        #   couldn't I use finish reason / stop reason to do this
        #    yes ... choices[0].finish_reason == "stop" would tell me when to mark it
        #    TODO what about finish_reason == "length" would it work too or is there a quirk in that case?
        #
        #
        # * last output chunk should be due to self-reporting "last" and not an extra output chunk from langchain
        chunk_with_finish_reason = output_chunks[-2]
        assert chunk_with_finish_reason is not None
        assert chunk_with_finish_reason.response_metadata is not None
        assert "finish_reason" in chunk_with_finish_reason.response_metadata
        output_finish_reason = chunk_with_finish_reason.response_metadata["finish_reason"]
        assert output_finish_reason in ("stop", "length")

        # FYI! STOP BLOCKING extra "last" chunk... clients can stop on metrics! (I always use metrics)
        #    or  have them stop on finish_reason if not using metrics
        # langchain adds this when the client doesn't self-report
        virtual_last_chunk = output_chunks[-1]
        assert virtual_last_chunk.chunk_position == "last"


class TestSGLANG_Streaming_Manually:

    # FYI run just one:     ptw -- -k "TestVLLM_Streaming_Manually and true" --capture no

    def test__when_stream_usage_is_true__finish_reason_and_stats_are_split_into_two_separate_chunks(self):
        # ptw -- -k "SGLANG and Streaming_Manually and false" --capture no

        print()  # avoid print on test name line

        model = ChatLlamaServer(
            api_key="",
            model="",
            base_url=SGLANG_QWEN38_BASE_URL,
            debugme=True,
            stream_usage=True,
            extra_body={
                # need these + stream_options.include_usage  (stream_usage enables this)
                "return_spec_tokens_details": True,
                "return_cached_tokens_details": True,
                # -- opt-in, custom final-SSE timing overlay (assume available for all local sglang deployments)
                "return_timings": True,
            })

        model.enable_tracking_input_chunks = True

        stream_chunks = model.stream(
            "what is 1+1? answer only",
            max_tokens=200,
        )
        output_chunks = [chunk for chunk in stream_chunks]

        # FYI server to server outputs chunk comparison:
        # - SGLANG "inputs" (server outputs) :
        #     -4 last_content chunk (content/reasoning_content)
        #     -3 finish_reason chunk (NOT with last content, after it)
        #     -2 sglext.spec_token_details+usage chunk (my custom extension is on its own SSE)
        #       btw langchain (or my provider: usage => usage_metadata (completely different field names)
        #     -1 timings.choices + usage chunk
        #   - TODO langchain outputs (document)
        # - VLLM "inputs":
        #     TODO double check this
        #     -2 last_content_chunk_has_finish_reason_too
        #     -1 stats_chunk (usage/metrics if include_usage)
        #       FYI downstream clients will need to accumulate stats (which makes sense as we do that with all SSEs to reconstruct original response)
        #   - TODO langchain outputs (document)
        # - llama-server "inputs":
        #     TODO verify (I think it matches vllm) llama-server's outputs:
        #     -2 lsat_content_chunk_has_finish_reason too???
        #     -1 timings chunk
        #   - TODO langchain outputs (document)

        # * INPUT ASSERTIONS (ON SGLANG's outputs)
        #  these are merely to troubleshoot server responses (aka inputs into langchain)
        #
        input_chunks = model._get_tracked_input_chunks()
        assert len(input_chunks) >= 1, "missing input chunks, should be 1+ input chunks"
        #
        last_input = input_chunks[-1]
        rich.print(f"{last_input=}")
        assert len(last_input["choices"]) == 0
        #
        second_to_last_input = input_chunks[-2]
        rich.print(f"{second_to_last_input=}")
        assert len(second_to_last_input["choices"]) == 1
        #
        # allow for model to ramble too much and get cut off with "length" seems to still behave the same as "stop"
        in_reason = second_to_last_input["choices"][0]["finish_reason"]
        assert in_reason in ("stop", "length")
        #
        assert second_to_last_input.get("usage") is None, "`usage` should be absent -- `stream_usage=False` was not requested"
        assert second_to_last_input.get("metrics") is None, "`metrics` should be absent -- `stream_usage=False` was not requested"
        #
        # VLLM notes:
        # - often multiple tokens per SSE (good idea IMO)

        # * OUTPUT ASSERTIONS
        #
        rich.print(output_chunks)
        assert len(output_chunks) >= 1, "output chunks should be 1 or more"
        assert len(output_chunks) == len(input_chunks)

        # * second to last has finish_reason
        #   as well as the last content/reasoning or both (not testing that here)
        chunk_with_finish_reason = output_chunks[-2]
        assert chunk_with_finish_reason is not None
        assert chunk_with_finish_reason.response_metadata is not None
        assert "finish_reason" in chunk_with_finish_reason.response_metadata
        output_finish_reason = chunk_with_finish_reason.response_metadata["finish_reason"]
        assert output_finish_reason in ("stop", "length")

        # * last chunk has metrics
        chunk_with_metrics = output_chunks[-1]
        assert "metrics" in chunk_with_metrics.response_metadata  # copied by _preserve_response_disagnostics
        assert "usage" in chunk_with_metrics.response_metadata  # copied by _preserve_response_disagnostics

        assert chunk_with_metrics.chunk_position == "last"

    def test__when_stream_usage_is_false(self):
        # ptw -- -k "SGLANG and Streaming_Manually and false" --capture no

        # TODO rename test for metrics too
        print()  # avoid print on test name line

        model = ChatLlamaServer(
            api_key="",
            model="",
            base_url=SGLANG_QWEN38_BASE_URL,
            debugme=True,
            stream_usage=False,
            # NOTE no extra_body
        )

        model.enable_tracking_input_chunks = True

        stream_chunks = model.stream(
            "what is 1+1? answer only",
            max_tokens=200,
        )
        output_chunks = [chunk for chunk in stream_chunks]

        # * INPUT ASSERTIONS (ON SGLANG's outputs)
        #  these are merely to troubleshoot server responses (aka inputs into langchain)
        #
        input_chunks = model._get_tracked_input_chunks()
        assert len(input_chunks) >= 1, "missing input chunks, should be 1+ input chunks"
        #
        last_input = input_chunks[-1]
        assert len(last_input["choices"]) == 1
        #
        # allow for model to ramble too much and get cut off with "length" seems to still behave the same as "stop"
        in_reason = last_input["choices"][0]["finish_reason"]
        assert in_reason in ("stop", "length")
        #
        assert last_input.get("usage") is None, "`usage` should be absent -- `stream_usage=False` was not requested"
        assert last_input.get("metrics") is None, "`metrics` should be absent -- `stream_usage=False` was not requested"
        #
        # VLLM notes:
        # - often multiple tokens per SSE (good idea IMO)

        # * OUTPUT ASSERTIONS
        #
        rich.print(output_chunks)
        assert len(output_chunks) >= 1, "output chunks should be 1 or more"
        assert len(output_chunks) == len(input_chunks) + 1
        # TODO do I care about marking last chunk so langchain doesn't add that for me as an extra chunk after last?
        #   in llama-server I use timings to do so... I could use metrics/usage to do so for vllm
        #   but that requires asking for usage or metrics (which I'd always be doing in my scenarios)
        #   couldn't I use finish reason / stop reason to do this
        #    yes ... choices[0].finish_reason == "stop" would tell me when to mark it
        #    TODO what about finish_reason == "length" would it work too or is there a quirk in that case?
        #
        #
        # * last output chunk should be due to self-reporting "last" and not an extra output chunk from langchain
        chunk_with_finish_reason = output_chunks[-2]
        assert chunk_with_finish_reason is not None
        assert chunk_with_finish_reason.response_metadata is not None
        assert "finish_reason" in chunk_with_finish_reason.response_metadata
        output_finish_reason = chunk_with_finish_reason.response_metadata["finish_reason"]
        assert output_finish_reason in ("stop", "length")

        # FYI! STOP BLOCKING extra "last" chunk... clients can stop on metrics! (I always use metrics)
        #    or  have them stop on finish_reason if not using metrics
        # langchain adds this when the client doesn't self-report
        virtual_last_chunk = output_chunks[-1]
        assert virtual_last_chunk.chunk_position == "last"
