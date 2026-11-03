"""Shared scopes restore their context even when resource cleanup fails."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

import pytest

from uncalled_for import Shared, SharedContext, resolved_dependencies


async def test_shared_cleanup_error_restores_unset_context() -> None:
    events: list[str] = []
    error = RuntimeError("cleanup failed")

    @asynccontextmanager
    async def healthy() -> AsyncGenerator[str]:
        events.append("healthy-enter")
        try:
            yield "healthy"
        finally:
            events.append("healthy-exit")

    @asynccontextmanager
    async def broken() -> AsyncGenerator[str]:
        events.append("broken-enter")
        try:
            yield "broken"
        finally:
            events.append("broken-exit")
            raise error

    async def handle(
        first: str = Shared(healthy), second: str = Shared(broken)
    ) -> None: ...

    assert SharedContext.stack.get(None) is None
    assert SharedContext.lock.get(None) is None
    assert SharedContext.resolved.get(None) is None

    with pytest.raises(RuntimeError) as caught:
        async with SharedContext(), resolved_dependencies(handle) as dependencies:
            assert dependencies == {"first": "healthy", "second": "broken"}

    assert caught.value is error
    assert events == ["healthy-enter", "broken-enter", "broken-exit", "healthy-exit"]
    with pytest.raises(LookupError):
        SharedContext.stack.get()
    with pytest.raises(LookupError):
        SharedContext.lock.get()
    with pytest.raises(LookupError):
        SharedContext.resolved.get()


async def test_shared_cleanup_error_restores_outer_scope() -> None:
    pool = object()
    closed = False
    error = RuntimeError("inner cleanup failed")

    @asynccontextmanager
    async def healthy() -> AsyncGenerator[object]:
        nonlocal closed
        try:
            yield pool
        finally:
            closed = True

    @asynccontextmanager
    async def broken() -> AsyncGenerator[str]:
        try:
            yield "inner"
        finally:
            raise error

    async def outer_handle(value: object = Shared(healthy)) -> None: ...
    async def inner_handle(value: str = Shared(broken)) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(outer_handle) as dependencies:
            assert dependencies["value"] is pool
        outer_stack = SharedContext.stack.get()
        outer_lock = SharedContext.lock.get()
        outer_resolved = SharedContext.resolved.get()

        with pytest.raises(RuntimeError) as caught:
            async with (
                SharedContext(),
                resolved_dependencies(inner_handle) as dependencies,
            ):
                assert dependencies["value"] == "inner"

        assert caught.value is error
        assert SharedContext.stack.get() is outer_stack
        assert SharedContext.lock.get() is outer_lock
        assert SharedContext.resolved.get() is outer_resolved
        async with resolved_dependencies(outer_handle) as dependencies:
            assert dependencies["value"] is pool
        assert not closed

    assert closed


@pytest.mark.parametrize("nested", [False, True])
async def test_shared_cleanup_cancellation_restores_context(nested: bool) -> None:
    cleanup_started = asyncio.Event()
    never_complete = asyncio.Event()
    restored = asyncio.Event()

    @asynccontextmanager
    async def resource() -> AsyncGenerator[str]:
        try:
            yield "pool"
        finally:
            cleanup_started.set()
            await never_complete.wait()

    async def handle(value: str = Shared(resource)) -> None: ...

    async def worker() -> None:
        async with AsyncExitStack() as scopes:
            if nested:
                await scopes.enter_async_context(SharedContext())
            previous_stack = SharedContext.stack.get(None)
            previous_lock = SharedContext.lock.get(None)
            previous_resolved = SharedContext.resolved.get(None)

            try:
                async with (
                    SharedContext(),
                    resolved_dependencies(handle) as dependencies,
                ):
                    assert dependencies["value"] == "pool"
            except asyncio.CancelledError:
                assert SharedContext.stack.get(None) is previous_stack
                assert SharedContext.lock.get(None) is previous_lock
                assert SharedContext.resolved.get(None) is previous_resolved
                restored.set()
                raise

    task = asyncio.create_task(worker())
    await cleanup_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert restored.is_set()
    assert task.cancelled()
