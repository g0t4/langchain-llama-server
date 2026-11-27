"""Calculate display statistics from preserved backend reports."""

from math import isfinite

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from .completion import AccumulatedCompletion, InterruptedCompletion


class InputStats(BaseModel):
    tokens: int | None = None
    cached_tokens: int | None = None
    tokens_per_second: float | None = None


class DraftStats(BaseModel):
    drafted_tokens: int | None = None
    accepted_tokens: int | None = None
    acceptance_rate: float | None = None
    mean_accept_length: float | None = None


class OutputStats(BaseModel):
    tokens: int | None = None
    reasoning_tokens: int | None = None
    non_reasoning_tokens: int | None = None
    tokens_per_second: float | None = None
    draft: DraftStats | None = None


class CompletionStats(BaseModel):
    input: InputStats
    output: OutputStats
    time_to_first_token_ms: float | None = None
    queue_ms: float | None = None


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _number(fields, key):
    value = fields.get(key)
    if type(value) in (float, int) and isfinite(value) and value >= 0:
        return value
    return None


def _count(fields, key):
    value = fields.get(key)
    return value if type(value) is int and value >= 0 else None


def _ratio(numerator, denominator):
    if numerator is not None and denominator is not None and denominator > 0:
        return numerator / denominator
    return None


def _draft(fields, drafted_key, accepted_key, rate_key=None, length_key=None):
    drafted = _count(fields, drafted_key)
    accepted = _count(fields, accepted_key)
    rate = _number(fields, rate_key)
    if rate is None and accepted is not None and drafted is not None and accepted <= drafted:
        rate = _ratio(accepted, drafted)
    length = _number(fields, length_key)
    if any(v is not None for v in (drafted, accepted, rate, length)):
        return DraftStats(drafted_tokens=drafted, accepted_tokens=accepted,
                          acceptance_rate=rate, mean_accept_length=length)
    return None


def calculate_stats(source: AccumulatedCompletion | InterruptedCompletion | AIMessage) -> CompletionStats:
    """Read the latest report of each kind. All calculations are request-local."""
    if isinstance(source, InterruptedCompletion):
        events = source.raw_chunks
    elif isinstance(source, AccumulatedCompletion):
        message = source.message
        events = source.raw_chunks or [message.response_metadata.get("raw_response", {})]
    else:
        metadata = source.response_metadata
        events = metadata.get("raw_chunks") or [metadata.get("raw_response", {})]
    reports = {}
    multiple_choices = False
    for event in events:
        for key in ("usage", "metrics", "sglext", "timings"):
            if isinstance(event.get(key), dict):
                reports[key] = event[key]
        multiple_choices |= any(c.get("index", 0) != 0 for c in event.get("choices") or [])

    usage = reports.get("usage", {})
    inp = InputStats(tokens=_count(usage, "prompt_tokens"),
                     cached_tokens=_count(_object(usage.get("prompt_tokens_details")), "cached_tokens"))
    out = OutputStats(tokens=_count(usage, "completion_tokens"),
                      reasoning_tokens=_count(_object(usage.get("completion_tokens_details")), "reasoning_tokens"))
    if out.reasoning_tokens is None:
        out.reasoning_tokens = _count(usage, "reasoning_tokens")

    # Usage is request-wide. Never combine it with a single choice's diagnostics.
    if multiple_choices:
        reports = {}
    timings = reports.get("timings", {})
    metrics = reports.get("metrics", {})
    ttft_ms = queue_ms = None
    if "choices" in timings or "sglext" in reports:
        choices = timings.get("choices") or []
        choice = choices[0] if len(choices) == 1 and choices[0].get("index", 0) == 0 else {}
        if inp.tokens is None:
            inp.tokens = _count(choice, "prompt_tokens")
        if inp.cached_tokens is None:
            inp.cached_tokens = _count(choice, "cached_tokens")
        if out.tokens is None:
            out.tokens = _count(choice, "completion_tokens")
        out.tokens_per_second = _number(choice, "decode_throughput")
        forward = _number(choice, "forward_entry_time")
        prefilled = _number(choice, "prefill_finished_time")
        if None not in (forward, prefilled, inp.tokens, inp.cached_tokens) and inp.tokens >= inp.cached_tokens:
            inp.tokens_per_second = _ratio(inp.tokens - inp.cached_tokens, prefilled - forward)
        ttft = _number(choice, "first_token_latency")
        queue = _number(choice, "queue_time")
        ttft_ms = ttft * 1000 if ttft is not None else None
        queue_ms = queue * 1000 if queue is not None else None
        spec = {**choice, **_object(reports.get("sglext", {}).get("spec_tokens_details"))}
        out.draft = _draft(spec, "spec_num_proposed_drafts", "spec_num_correct_drafts",
                           "spec_accept_rate", "spec_accept_length")
    elif metrics:
        ttft_ms = _number(metrics, "time_to_first_token_ms")
        queue_ms = _number(metrics, "queue_time_ms")
        out.tokens_per_second = _ratio(1000, _number(metrics, "mean_itl_ms"))
        out.draft = _draft(_object(metrics.get("speculative_decoding")), "num_draft_tokens",
                           "num_accepted_draft_tokens", "draft_acceptance_rate", "mean_acceptance_length")
    elif timings:
        if inp.cached_tokens is None:
            inp.cached_tokens = _count(timings, "cache_n")
        evaluated = _count(timings, "prompt_n")
        if inp.tokens is None and evaluated is not None and inp.cached_tokens is not None:
            inp.tokens = evaluated + inp.cached_tokens
        if out.tokens is None:
            out.tokens = _count(timings, "predicted_n")
        inp.tokens_per_second = _number(timings, "prompt_per_second")
        out.tokens_per_second = _number(timings, "predicted_per_second")
        out.draft = _draft(timings, "draft_n", "draft_n_accepted")

    if out.reasoning_tokens is not None and out.tokens is not None and out.reasoning_tokens <= out.tokens:
        out.non_reasoning_tokens = out.tokens - out.reasoning_tokens
    return CompletionStats(input=inp, output=out, time_to_first_token_ms=ttft_ms, queue_ms=queue_ms)
