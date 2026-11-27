from collections.abc import AsyncIterator, Iterator

from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message
from pydantic import BaseModel


class AccumulatedCompletion(BaseModel):
    """A finished assistant message and the ordered backend events that produced it."""

    message: AIMessage
    raw_chunks: list[dict]


class _Accumulator:
    def __init__(self):
        self.raw_chunks: list[dict] = []
        self._parts: list[AIMessage] = []
        self._result: AccumulatedCompletion | None = None

    def _record(self, chunk: AIMessage) -> None:
        # Keep a snapshot before handing the editable chunk to the consumer.
        part = chunk.model_copy(deep=True)
        self.raw_chunks.extend(part.response_metadata.pop("raw_chunks", []))
        self._parts.append(part)

    def _finish(self) -> None:
        if not self._parts:
            raise ValueError("The stream ended without an assistant message.")
        first, *rest = self._parts
        # LangChain yields one ordinary AIMessage when streaming is disabled.
        if not isinstance(first, AIMessageChunk):
            self._result = AccumulatedCompletion(message=first, raw_chunks=self.raw_chunks)
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
        async for chunk in chunks:
            self._record(chunk)
            yield chunk
        self._finish()

    async def __anext__(self):
        return await anext(self._iterator)

    async def aclose(self):
        await self._iterator.aclose()
