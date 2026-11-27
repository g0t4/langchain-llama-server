from asyncio import CancelledError
from contextlib import asynccontextmanager, contextmanager
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from time import monotonic_ns
from typing import Literal

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

    outcome: Literal["completed"] = "completed"
    message: AIMessage
    raw_chunks: list[dict]
    client_timing: ClientTiming | None = None


class InterruptedCompletion(BaseModel):
    """Captured backend events from a stream that did not naturally exhaust."""

    outcome: Literal["cancelled", "error"]
    raw_chunks: list[dict]
    client_timing: ClientTiming | None
    backend: str | None = None
    model: str | None = None
    headers: dict[str, str] | None = None
    error_type: str | None = None
    error_message: str | None = None


class _Accumulator:
    def __init__(self):
        self.raw_chunks: list[dict] = []
        self._parts: list[AIMessage] = []
        self._result: AccumulatedCompletion | InterruptedCompletion | None = None

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

    def _end_timing(self) -> None:
        if self.client_timing is None or self.client_timing.ended_at is not None:
            return
        self.client_timing.elapsed_ms = self._elapsed_ms()
        self.client_timing.ended_at = datetime.now(timezone.utc)

    def _interrupt(self, error: BaseException | None = None) -> None:
        if self._result is not None:
            return
        self._end_timing()
        cancelled = error is None or isinstance(error, (GeneratorExit, CancelledError, KeyboardInterrupt))
        metadata = {}
        for part in self._parts:
            for key in ("backend", "model", "headers"):
                if key in part.response_metadata:
                    metadata[key] = part.response_metadata[key]
        self._result = InterruptedCompletion(
            outcome="cancelled" if cancelled else "error",
            raw_chunks=self.raw_chunks, client_timing=self.client_timing,
            **metadata,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )
        self._parts.clear()

    def _finish(self) -> None:
        self._end_timing()
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
        # Retain exact arguments for consumers that strictly validate completed calls.
        if merged.tool_call_chunks:
            merged.additional_kwargs["tool_calls"] = [
                {"id": call["id"], "type": "function",
                 "function": {"name": call["name"], "arguments": call["args"]}}
                for call in sorted(merged.tool_call_chunks, key=lambda c: c["index"] or 0)
            ]
        self._result = AccumulatedCompletion(
            message=message_chunk_to_message(merged), raw_chunks=self.raw_chunks,
            client_timing=self.client_timing,
        )
        self._parts.clear()

    def get_trace(self) -> AccumulatedCompletion | InterruptedCompletion:
        if self._result is None:
            raise RuntimeError("The stream is still open.")
        return self._result

    def get_final_completion(self) -> AccumulatedCompletion:
        if not isinstance(self._result, AccumulatedCompletion):
            raise RuntimeError("A final completion requires a naturally exhausted stream.")
        return self._result


class CompletionStream(_Accumulator, Iterator[AIMessageChunk]):
    def __init__(self, chunks: Iterator[AIMessageChunk]):
        super().__init__()
        self._chunks = chunks
        self._iterator = self._capture(chunks)

    def _capture(self, chunks):
        self._start()
        try:
            for chunk in chunks:
                self._record(chunk)
                yield chunk
            self._finish()
        except BaseException as error:
            self._interrupt(error)
            raise

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self._interrupt()
        self._iterator.close()
        if close := getattr(self._chunks, "close", None):
            close()


class AsyncCompletionStream(_Accumulator, AsyncIterator[AIMessageChunk]):
    def __init__(self, chunks: AsyncIterator[AIMessageChunk]):
        super().__init__()
        self._chunks = chunks
        self._iterator = self._capture(chunks)

    async def _capture(self, chunks):
        self._start()
        try:
            async for chunk in chunks:
                self._record(chunk)
                yield chunk
            self._finish()
        except BaseException as error:
            self._interrupt(error)
            raise

    async def __anext__(self):
        return await anext(self._iterator)

    async def aclose(self):
        self._interrupt()
        await self._iterator.aclose()
        if close := getattr(self._chunks, "aclose", None):
            await close()


@contextmanager
def capture_completion(chunks: Iterator[AIMessageChunk]) -> Iterator[CompletionStream]:
    """Capture an existing model or tool-bound stream, closing it on early exit."""
    stream = CompletionStream(chunks)
    try:
        yield stream
    except BaseException as error:
        stream._interrupt(error)
        raise
    finally:
        stream.close()


@asynccontextmanager
async def acapture_completion(chunks: AsyncIterator[AIMessageChunk]) -> AsyncIterator[AsyncCompletionStream]:
    """Async counterpart to capture_completion."""
    stream = AsyncCompletionStream(chunks)
    try:
        yield stream
    except BaseException as error:
        stream._interrupt(error)
        raise
    finally:
        await stream.aclose()
