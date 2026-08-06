"""Tests for resolution frames."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import cast

import pytest

from uncalled_for import CycleError, Dependency, current_frame, frame_scope
from uncalled_for.functional import _Depends  # pyright: ignore[reportPrivateUsage]


class _Counting(Dependency[str]):
    """Counts how many times the resolution engine enters it."""

    def __init__(self) -> None:
        self.enters = 0

    async def __aenter__(self) -> str:
        self.enters += 1
        return f"value-{self.enters}"


class _Exploding(Dependency[str]):
    """Fails on entry, and counts the attempts."""

    def __init__(self) -> None:
        self.enters = 0

    async def __aenter__(self) -> str:
        self.enters += 1
        raise RuntimeError("boom")


class _Recording(Dependency[str]):
    """Records the order of for_parameter and __aenter__ calls."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def for_parameter(self, name: str) -> Dependency[str]:
        self.events.append(f"for_parameter:{name}")
        return self

    async def __aenter__(self) -> str:
        self.events.append("aenter")
        return "recorded"


def test_frame_scope_sets_and_resets_the_current_frame() -> None:
    def target() -> None: ...

    with frame_scope(target) as frame:
        assert current_frame() is frame
        assert frame.function is target

    with pytest.raises(RuntimeError):
        current_frame()


def test_current_frame_without_a_scope_raises() -> None:
    with pytest.raises(RuntimeError, match="frame_scope"):
        current_frame()


def test_cycle_error_is_a_value_error() -> None:
    assert isinstance(CycleError("nope"), ValueError)


async def test_resolve_returns_a_provided_value() -> None:
    def target(value: str = cast(str, _Counting())) -> None: ...

    with frame_scope(target, {"value": "given"}) as frame:
        assert await frame.resolve("value") == "given"


async def test_resolve_enters_a_dependency_once_and_memoizes_it() -> None:
    dependency = _Counting()

    def target(value: str = cast(str, dependency)) -> None: ...

    async with AsyncExitStack() as stack:
        _Depends.cache.set({})
        _Depends.stack.set(stack)
        with frame_scope(target) as frame:
            assert await frame.resolve("value") == "value-1"
            assert await frame.resolve("value") == "value-1"

    assert dependency.enters == 1


async def test_resolve_of_an_unknown_name_raises() -> None:
    def target() -> None: ...

    with frame_scope(target) as frame:
        with pytest.raises(LookupError) as caught:
            await frame.resolve("missing")

    assert "target" in str(caught.value)
    assert "missing" in str(caught.value)


async def test_resolve_of_an_unsupplied_required_parameter_raises() -> None:
    def target(value: str) -> None: ...

    with frame_scope(target) as frame:
        with pytest.raises(LookupError) as caught:
            await frame.resolve("value")

    assert "received no value" in str(caught.value)
    assert "value" in str(caught.value)


async def test_resolve_returns_a_plain_signature_default() -> None:
    def target(limit: int = 10) -> None: ...

    with frame_scope(target) as frame:
        assert await frame.resolve("limit") == 10


async def test_provided_only_returns_a_provided_value() -> None:
    def target(value: str = cast(str, _Counting())) -> None: ...

    with frame_scope(target, {"value": "given"}) as frame:
        assert await frame.resolve("value", provided_only=True) == "given"


async def test_provided_only_never_enters_a_dependency() -> None:
    dependency = _Counting()

    def target(value: str = cast(str, dependency)) -> None: ...

    async with AsyncExitStack() as stack:
        _Depends.cache.set({})
        _Depends.stack.set(stack)
        with frame_scope(target) as frame:
            with pytest.raises(LookupError) as caught:
                await frame.resolve("value", provided_only=True)
            assert dependency.enters == 0

    assert "target" in str(caught.value)
    assert "value" in str(caught.value)


async def test_provided_only_ignores_a_plain_signature_default() -> None:
    def target(limit: int = 10) -> None: ...

    with frame_scope(target) as frame:
        with pytest.raises(LookupError):
            await frame.resolve("limit", provided_only=True)


async def test_resolve_memoizes_a_failure() -> None:
    dependency = _Exploding()

    def target(value: str = cast(str, dependency)) -> None: ...

    async with AsyncExitStack() as stack:
        _Depends.cache.set({})
        _Depends.stack.set(stack)
        with frame_scope(target) as frame:
            with pytest.raises(RuntimeError) as first:
                await frame.resolve("value")
            with pytest.raises(RuntimeError) as second:
                await frame.resolve("value")
            assert dependency.enters == 1

    assert first.value is second.value


async def test_resolve_calls_for_parameter_before_entering() -> None:
    dependency = _Recording()

    def target(value: str = cast(str, dependency)) -> None: ...

    async with AsyncExitStack() as stack:
        _Depends.cache.set({})
        _Depends.stack.set(stack)
        with frame_scope(target) as frame:
            assert await frame.resolve("value") == "recorded"

    assert dependency.events == ["for_parameter:value", "aenter"]


async def test_a_self_reference_raises_a_cycle_error() -> None:
    def target(x: str = cast(str, _Counting())) -> None: ...

    with frame_scope(target) as frame:
        frame.resolving.append("x")
        with pytest.raises(CycleError) as caught:
            await frame.resolve("x")

    assert "target" in str(caught.value)
    assert "x -> x" in str(caught.value)
