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

    with pytest.raises(ValueError, match="Only one _SingleDep dependency is allowed"):
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


def test_duplicate_concrete_type_names_the_concrete_type() -> None:
    """Two instances of the same single type: error names the concrete type,
    not an ancestor, and is deterministic across runs."""

    class Handler(Dependency["Handler"]):
        single = True

        async def __aenter__(self) -> Handler: ...

    class Retry(Handler):
        async def __aenter__(self) -> Retry: ...

    async def my_func(
        a: Retry = Retry(),
        b: Retry = Retry(),
    ) -> None: ...

    with pytest.raises(ValueError, match="Only one Retry dependency is allowed$"):
        validate_dependencies(my_func)


def test_different_subclasses_of_single_base_names_the_base() -> None:
    """Two different subclasses of a single base: error names the shared
    base and lists both concrete types."""

    class Runtime(Dependency["Runtime"]):
        single = True

        async def __aenter__(self) -> Runtime: ...

    class Timeout(Runtime):
        async def __aenter__(self) -> Timeout: ...

    class Deadline(Runtime):
        async def __aenter__(self) -> Deadline: ...

    async def my_func(
        a: Timeout = Timeout(),
        b: Deadline = Deadline(),
    ) -> None: ...

    with pytest.raises(
        ValueError,
        match=r"Only one Runtime dependency is allowed, but found: .+",
    ):
        validate_dependencies(my_func)


def test_no_dependencies_is_valid() -> None:
    async def my_func(x: int, y: str) -> None: ...

    validate_dependencies(my_func)
