"""Tests for the Shared() app-scoped dependency."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from uncalled_for import Depends, Shared, SharedContext, resolved_dependencies


async def test_shared_resolves_once_across_calls() -> None:
    call_count = 0

    def make_config() -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        return {"url": "http://example.com"}

    async def func_a(config: dict[str, str] = Shared(make_config)) -> None: ...
    async def func_b(config: dict[str, str] = Shared(make_config)) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(func_a) as deps_a:
            assert deps_a["config"] == {"url": "http://example.com"}

        async with resolved_dependencies(func_b) as deps_b:
            assert deps_b["config"] == {"url": "http://example.com"}

        assert call_count == 1


async def test_shared_async_context_manager_lifecycle() -> None:
    entered = False
    exited = False

    @asynccontextmanager
    async def make_pool() -> AsyncGenerator[str]:
        nonlocal entered, exited
        entered = True
        yield "pool-connection"
        exited = True

    async def my_func(pool: str = Shared(make_pool)) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(my_func) as deps:
            assert deps["pool"] == "pool-connection"
            assert entered
            assert not exited

    assert exited


async def test_shared_with_nested_depends() -> None:
    def get_host() -> str:
        return "localhost"

    def get_url(host: str = Depends(get_host)) -> str:
        return f"http://{host}:5432"

    async def my_func(url: str = Shared(get_url)) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(my_func) as deps:
            assert deps["url"] == "http://localhost:5432"


async def test_shared_identity_is_factory_function() -> None:
    def make_value() -> int:
        return 42

    async def func_a(v: int = Shared(make_value)) -> None: ...
    async def func_b(v: int = Shared(make_value)) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(func_a) as deps_a:
            pass
        async with resolved_dependencies(func_b) as deps_b:
            pass

        assert deps_a["v"] is deps_b["v"]


async def test_shared_async_function() -> None:
    async def make_value() -> str:
        return "async-shared"

    async def my_func(v: str = Shared(make_value)) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(my_func) as deps:
            assert deps["v"] == "async-shared"


async def test_shared_context_cleanup_order() -> None:
    order: list[str] = []

    @asynccontextmanager
    async def resource_a() -> AsyncGenerator[str]:
        order.append("a-enter")
        yield "a"
        order.append("a-exit")

    @asynccontextmanager
    async def resource_b() -> AsyncGenerator[str]:
        order.append("b-enter")
        yield "b"
        order.append("b-exit")

    async def func_a(v: str = Shared(resource_a)) -> None: ...
    async def func_b(v: str = Shared(resource_b)) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(func_a) as deps:
            assert deps["v"] == "a"
        async with resolved_dependencies(func_b) as deps:
            assert deps["v"] == "b"

    assert order == ["a-enter", "b-enter", "b-exit", "a-exit"]
