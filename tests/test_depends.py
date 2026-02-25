"""Tests for the Depends() functional dependency."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from uncalled_for import Depends, resolved_dependencies


async def test_sync_function() -> None:
    def get_value() -> str:
        return "sync"

    async def my_func(v: str = Depends(get_value)) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert deps["v"] == "sync"


async def test_async_function() -> None:
    async def get_value() -> str:
        return "async"

    async def my_func(v: str = Depends(get_value)) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert deps["v"] == "async"


async def test_sync_context_manager() -> None:
    cleanup_called = False

    @contextmanager
    def get_value() -> Generator[str, None, None]:
        nonlocal cleanup_called
        yield "sync-cm"
        cleanup_called = True

    async def my_func(v: str = Depends(get_value)) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert deps["v"] == "sync-cm"
        assert not cleanup_called

    assert cleanup_called


async def test_async_context_manager() -> None:
    cleanup_called = False

    @asynccontextmanager
    async def get_value() -> AsyncGenerator[str]:
        nonlocal cleanup_called
        yield "async-cm"
        cleanup_called = True

    async def my_func(v: str = Depends(get_value)) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert deps["v"] == "async-cm"
        assert not cleanup_called

    assert cleanup_called


async def test_dependency_caching_within_scope() -> None:
    call_count = 0

    def expensive() -> int:
        nonlocal call_count
        call_count += 1
        return 42

    async def my_func(
        a: int = Depends(expensive),
        b: int = Depends(expensive),
    ) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert deps["a"] == 42
        assert deps["b"] == 42
        assert call_count == 1


async def test_nested_dependencies() -> None:
    def get_base() -> str:
        return "base"

    def get_derived(base: str = Depends(get_base)) -> str:
        return f"{base}-derived"

    async def my_func(v: str = Depends(get_derived)) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert deps["v"] == "base-derived"


async def test_kwargs_override() -> None:
    def get_value() -> str: ...

    async def my_func(v: str = Depends(get_value)) -> None: ...

    async with resolved_dependencies(my_func, {"v": "override"}) as deps:
        assert deps["v"] == "override"


async def test_non_dependency_parameters_ignored() -> None:
    def get_value() -> str:
        return "injected"

    async def my_func(
        regular: str,
        v: str = Depends(get_value),
        also_regular: int = 5,
    ) -> Any: ...

    async with resolved_dependencies(my_func) as deps:
        assert deps == {"v": "injected"}
