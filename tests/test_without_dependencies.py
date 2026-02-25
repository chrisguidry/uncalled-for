"""Tests for without_dependencies()."""

from __future__ import annotations

import inspect
from typing import Any

from uncalled_for import Dependency, Depends, without_dependencies


async def test_no_dependencies_returns_original() -> None:
    async def plain(x: int, y: str) -> str:
        return f"{x}-{y}"

    result = without_dependencies(plain)
    assert result is plain
    assert await result(x=1, y="a") == "1-a"


async def test_dependency_params_excluded_from_signature() -> None:
    def get_db() -> str:
        return "db"

    async def handler(name: str, db: str = Depends(get_db)) -> str:
        return f"{name}:{db}"

    wrapper = without_dependencies(handler)
    sig = inspect.signature(wrapper)

    assert "name" in sig.parameters
    assert "db" not in sig.parameters

    result = await wrapper(name="x")
    assert result == "x:db"


async def test_resolves_async_factory() -> None:
    async def get_value() -> str:
        return "async-resolved"

    async def handler(v: str = Depends(get_value)) -> str:
        return v

    wrapper = without_dependencies(handler)
    result = await wrapper()

    assert result == "async-resolved"


async def test_resolves_sync_factory() -> None:
    def get_value() -> str:
        return "sync-resolved"

    async def handler(v: str = Depends(get_value)) -> str:
        return v

    wrapper = without_dependencies(handler)
    result = await wrapper()

    assert result == "sync-resolved"


async def test_wraps_sync_handler() -> None:
    def get_value() -> str:
        return "injected"

    def handler(v: str = Depends(get_value)) -> str:
        return v

    wrapper = without_dependencies(handler)
    result = await wrapper()

    assert result == "injected"


class CustomDep(Dependency[str]):
    async def __aenter__(self) -> str:
        return "custom"


async def test_works_with_dependency_subclass() -> None:
    async def handler(v: str = CustomDep()) -> str:  # type: ignore[assignment]
        return v

    wrapper = without_dependencies(handler)
    sig = inspect.signature(wrapper)

    assert "v" not in sig.parameters

    result = await wrapper()
    assert result == "custom"


async def test_preserves_name_and_doc() -> None:
    def get_db() -> str:
        return "db"

    async def my_handler(name: str, db: str = Depends(get_db)) -> str:
        """Handler docstring."""
        return f"{name}:{db}"

    wrapper = without_dependencies(my_handler)

    assert wrapper.__name__ == "my_handler"
    assert wrapper.__doc__ == "Handler docstring."

    result = await wrapper(name="test")
    assert result == "test:db"


async def test_user_kwargs_passed_through() -> None:
    def get_db() -> str:
        return "db-conn"

    async def handler(
        name: str, count: int, db: str = Depends(get_db)
    ) -> dict[str, Any]:
        return {"name": name, "count": count, "db": db}

    wrapper = without_dependencies(handler)
    result = await wrapper(name="alice", count=3)

    assert result == {"name": "alice", "count": 3, "db": "db-conn"}


async def test_cached() -> None:
    def get_db() -> str:
        return "db"

    async def handler(db: str = Depends(get_db)) -> str:
        return db

    assert without_dependencies(handler) is without_dependencies(handler)

    result = await without_dependencies(handler)()
    assert result == "db"
