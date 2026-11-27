from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from time import monotonic_ns

from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message
from pydantic import BaseModel, Field, computed_field


class ClientTiming(BaseModel):
    """UTC wall times and monotonic milliseconds observed by the stream consumer."""

    started_at: datetime
    ended_at: datetime | None = None
    elapsed_ms: float | None = None
    event_offsets_ms: list[float] = Field(default_factory=list)
    first_reasoning_ms: float | None = None
    first_answer_ms: float | None = None

    @computed_field
    @property
    def first_event_ms(self) -> float | None:
        return self.event_offsets_ms[0] if self.event_offsets_ms else None


class AccumulatedCompletion(BaseModel):
    """A finished assistant message and the ordered backend events that produced it."""

    message: AIMessage
    raw_chunks: list[dict]
    client_timing: ClientTiming | None = None


class _Accumulator:
    def __init__(self):
        self.raw_chunks: list[dict] = []
        self._parts: list[AIMessage] = []
        self._result: AccumulatedCompletion | None = None

        self.client_timing: ClientTiming | None = None

    def _start(self) -> None:
        self._started_ns = monotonic_ns()
        self.client_timing = ClientTiming(started_at=datetime.now(timezone.utc))

    def _elapsed_ms(self) -> float:
        return (monotonic_ns() - self._started_ns) / 1_000_000

    def _record(self, chunk: AIMessage) -> None:
        offset = self._elapsed_ms()
        timing = self.client_timing
        assert timing is not None
        events = chunk.response_metadata.get("raw_chunks", [])
        timing.event_offsets_ms.extend([offset] * len(events))
        if timing.first_reasoning_ms is None and chunk.additional_kwargs.get("reasoning_content"):
            timing.first_reasoning_ms = offset
        if timing.first_answer_ms is None and chunk.text:
            timing.first_answer_ms = offset

        # Keep a snapshot before handing the editable chunk to the consumer.
        part = chunk.model_copy(deep=True)
        self.raw_chunks.extend(part.response_metadata.pop("raw_chunks", []))
        self._parts.append(part)

    def _finish(self) -> None:
        assert self.client_timing is not None
        self.client_timing.elapsed_ms = self._elapsed_ms()
        self.client_timing.ended_at = datetime.now(timezone.utc)
        if not self._parts:
            raise ValueError("The stream ended without an assistant message.")
        first, *rest = self._parts
        # LangChain yields one ordinary AIMessage when streaming is disabled.
        if not isinstance(first, AIMessageChunk):
            self._result = AccumulatedCompletion(
                message=first, raw_chunks=self.raw_chunks, client_timing=self.client_timing,
            )
            self._parts.clear()
            return
        # Usage reports on these backends are cumulative snapshots.
        usage = next((p.usage_metadata for p in reversed(self._parts)
                      if p.usage_metadata is not None), None)
        for part in self._parts:
            part.usage_metadata = None
        merged = first + rest
        merged.usage_metadata = usage
        self._result = AccumulatedCompletion(
            message=message_chunk_to_message(merged), raw_chunks=self.raw_chunks,
            client_timing=self.client_timing,
        )
        self._parts.clear()

    def get_final_completion(self) -> AccumulatedCompletion:
        if self._result is None:
            raise RuntimeError("A final completion requires a naturally exhausted stream.")
        return self._result


class CompletionStream(_Accumulator, Iterator[AIMessageChunk]):
    def __init__(self, chunks: Iterator[AIMessageChunk]):
        super().__init__()
        self._iterator = self._capture(chunks)

    def _capture(self, chunks):
        self._start()
        for chunk in chunks:
            self._record(chunk)
            yield chunk
        self._finish()

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self._iterator.close()


class AsyncCompletionStream(_Accumulator, AsyncIterator[AIMessageChunk]):
    def __init__(self, chunks: AsyncIterator[AIMessageChunk]):
        super().__init__()
        self._iterator = self._capture(chunks)

    async def _capture(self, chunks):
        self._start()
        async for chunk in chunks:
            self._record(chunk)
            yield chunk
        self._finish()

    async def __anext__(self):
        return await anext(self._iterator)

    async def aclose(self):
        await self._iterator.aclose()
