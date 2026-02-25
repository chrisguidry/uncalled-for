"""Tests for dependency validation."""

from __future__ import annotations

from typing import cast

import pytest

from uncalled_for import Dependency, validate_dependencies


class _SingleDep(Dependency[str]):
    single = True

    async def __aenter__(self) -> str: ...


def SingleDep() -> str:
    return cast(str, _SingleDep())


class _NotSingle(Dependency[str]):
    async def __aenter__(self) -> str: ...


def NotSingle() -> str:
    return cast(str, _NotSingle())


def test_valid_no_conflicts() -> None:
    async def my_func(
        a: str = NotSingle(),
        b: str = NotSingle(),
    ) -> None: ...

    validate_dependencies(my_func)


def test_single_dep_alone_is_valid() -> None:
    async def my_func(a: str = SingleDep()) -> None: ...

    validate_dependencies(my_func)


def test_duplicate_single_type_raises() -> None:
    async def my_func(
        a: str = SingleDep(),
        b: str = SingleDep(),
    ) -> None: ...

    with pytest.raises(ValueError, match="Only one _SingleDep"):
        validate_dependencies(my_func)


def test_single_base_class_conflict() -> None:
    class _Runtime(Dependency[str]):
        single = True

        async def __aenter__(self) -> str: ...

    class _TimeoutRuntime(_Runtime):
        async def __aenter__(self) -> str: ...

    class _DeadlineRuntime(_Runtime):
        async def __aenter__(self) -> str: ...

    def TimeoutRuntime() -> str:
        return cast(str, _TimeoutRuntime())

    def DeadlineRuntime() -> str:
        return cast(str, _DeadlineRuntime())

    async def my_func(
        a: str = TimeoutRuntime(),
        b: str = DeadlineRuntime(),
    ) -> None: ...

    with pytest.raises(ValueError, match="Only one _Runtime"):
        validate_dependencies(my_func)


def test_no_dependencies_is_valid() -> None:
    async def my_func(x: int, y: str) -> None: ...

    validate_dependencies(my_func)
