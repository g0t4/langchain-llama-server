from langchain_core.messages import AIMessage
import pytest

from langchain_openai_compatible import AccumulatedCompletion, calculate_stats


def completion(*events):
    return AccumulatedCompletion(message=AIMessage(content=""), raw_chunks=list(events))


def test_recalculate_split_sglang_reports_from_saved_trace():
    trace = completion(
        {"sglext": {"spec_tokens_details": {"spec_accept_length": 3.5,
                                            "spec_num_proposed_drafts": 12,
                                            "spec_num_correct_drafts": 6}}},
        {"usage": {"completion_tokens": 8}},
        {"usage": {"prompt_tokens": 60, "completion_tokens": 10, "reasoning_tokens": 0},
         "timings": {"choices": [{"index": 0, "cached_tokens": 20, "forward_entry_time": 10,
                                  "prefill_finished_time": 10.25, "decode_throughput": 50,
                                  "queue_time": 0, "first_token_latency": 0.5}]}},
    )
    saved = trace.model_dump_json()
    stats = calculate_stats(AccumulatedCompletion.model_validate_json(saved))
    assert stats.input.model_dump() == {"tokens": 60, "cached_tokens": 20, "tokens_per_second": 160}
    assert stats.output.tokens == stats.output.non_reasoning_tokens == 10
    assert stats.output.tokens_per_second == 50
    assert stats.output.draft.acceptance_rate == 0.5
    assert stats.output.draft.mean_accept_length == 3.5
    assert stats.time_to_first_token_ms == 500
    assert stats.queue_ms == 0
    assert trace.model_dump_json() == saved


def test_vllm_json_uses_decode_interval_and_reported_draft_rate():
    payload = {"usage": {"completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 5}},
               "metrics": {"tokens_per_second": 10, "mean_itl_ms": 5, "time_to_first_token_ms": 300,
                           "speculative_decoding": {"num_draft_tokens": 20, "num_accepted_draft_tokens": 5,
                                                    "draft_acceptance_rate": 0.5, "mean_acceptance_length": 3}}}
    message = AIMessage(content="answer", response_metadata={"raw_response": payload})
    stats = calculate_stats(message)
    assert stats.output.tokens_per_second == 200
    assert stats.input.tokens_per_second is None
    assert stats.output.non_reasoning_tokens == 15
    assert stats.output.draft.acceptance_rate == 0.5
    assert stats.time_to_first_token_ms == 300
    assert stats == calculate_stats(AccumulatedCompletion(message=message, raw_chunks=[]))


def test_llama_server_timings_without_usage():
    stats = calculate_stats(completion({"timings": {"cache_n": 12, "prompt_n": 3, "predicted_n": 9,
                                                    "prompt_per_second": 250, "predicted_per_second": 40,
                                                    "draft_n": 0, "draft_n_accepted": 0}}))
    assert stats.input.tokens == 15
    assert stats.input.cached_tokens == 12
    assert stats.input.tokens_per_second == 250
    assert stats.output.tokens == 9
    assert stats.output.tokens_per_second == 40
    assert stats.output.draft.drafted_tokens == 0
    assert stats.output.draft.acceptance_rate is None
    assert stats.output.non_reasoning_tokens is None


@pytest.mark.parametrize("reasoning", [None, 21])
def test_missing_and_inconsistent_counts(reasoning):
    stats = calculate_stats(completion({"usage": {"completion_tokens": 20, "reasoning_tokens": reasoning},
                                        "metrics": {"mean_itl_ms": 0}}))
    assert stats.output.reasoning_tokens == reasoning
    assert stats.output.non_reasoning_tokens is None
    assert stats.output.tokens_per_second is None
    assert stats.output.draft is None


def test_multiple_choices_preserve_request_counts():
    stats = calculate_stats(completion({"choices": [{"index": 0}, {"index": 1}],
                                        "usage": {"completion_tokens": 8},
                                        "timings": {"predicted_per_second": 7}}))
    assert stats.output.tokens == 8
    assert stats.output.tokens_per_second is None
