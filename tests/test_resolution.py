"""Tests for resolved_dependencies and FailedDependency."""

from __future__ import annotations
from typing import cast

from uncalled_for import Dependency, FailedDependency, resolved_dependencies


class _Boom(Dependency[str]):
    async def __aenter__(self) -> str:
        raise RuntimeError("kaboom")


def Boom() -> str:
    return cast(str, _Boom())


async def test_failed_dependency_captured() -> None:
    async def my_func(v: str = Boom()) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert isinstance(deps["v"], FailedDependency)
        assert deps["v"].parameter == "v"
        assert isinstance(deps["v"].error, RuntimeError)
        assert str(deps["v"].error) == "kaboom"


async def test_empty_function_resolves_empty() -> None:
    async def my_func() -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert deps == {}


class _Simple(Dependency[str]):
    async def __aenter__(self) -> str:
        return "injected"


def Simple() -> str:
    return cast(str, _Simple())


async def test_mixed_deps_and_kwargs() -> None:
    async def my_func(
        a: str = Simple(),
        b: str = Simple(),
    ) -> None: ...

    async with resolved_dependencies(my_func, {"a": "provided"}) as deps:
        assert deps["a"] == "provided"
        assert deps["b"] == "injected"
