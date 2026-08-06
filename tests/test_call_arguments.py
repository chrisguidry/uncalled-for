"""Tests for the CallArgument() dependency."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import cast

import pytest

from uncalled_for import (
    CallArgument,
    CycleError,
    Dependency,
    Depends,
    FailedDependency,
    resolved_dependencies,
)
from uncalled_for.frames import _CallArgument  # pyright: ignore[reportPrivateUsage]
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


class _Quiet(_CallArgument):
    """A subclass used to check that for_parameter keeps the runtime type."""


async def test_a_bare_call_argument_reads_the_matching_argument() -> None:
    def load(user_id: str = CallArgument()) -> str:
        return f"user:{user_id}"

    async def handle(user_id: str, profile: str = Depends(load)) -> None: ...

    async with resolved_dependencies(handle, {"user_id": "u-1"}) as deps:
        assert deps["profile"] == "user:u-1"


async def test_a_named_call_argument_reads_another_parameter() -> None:
    def load(account: str = CallArgument("user_id")) -> str:
        return f"user:{account}"

    async def handle(user_id: str, profile: str = Depends(load)) -> None: ...

    async with resolved_dependencies(handle, {"user_id": "u-2"}) as deps:
        assert deps["profile"] == "user:u-2"


async def test_a_call_argument_reads_a_dependency_backed_parameter() -> None:
    calls = 0

    def get_destination() -> str:
        nonlocal calls
        calls += 1
        return "moon"

    def make_ticket(destination: str = CallArgument()) -> str:
        return f"ticket to {destination}"

    async def travel(
        ticket: str = Depends(make_ticket),
        destination: str = Depends(get_destination),
    ) -> None: ...

    async with resolved_dependencies(travel) as deps:
        assert deps["ticket"] == "ticket to moon"
        assert deps["destination"] == "moon"

    assert calls == 1


async def test_a_caller_override_wins_over_the_backing_dependency() -> None:
    def get_destination() -> str: ...

    def make_ticket(destination: str = CallArgument()) -> str:
        return f"ticket to {destination}"

    async def travel(
        ticket: str = Depends(make_ticket),
        destination: str = Depends(get_destination),
    ) -> None: ...

    async with resolved_dependencies(travel, {"destination": "mars"}) as deps:
        assert deps["ticket"] == "ticket to mars"
        assert deps["destination"] == "mars"


async def test_an_optional_call_argument_yields_none_when_absent() -> None:
    def load(tenant: str | None = CallArgument(optional=True)) -> str:
        return f"tenant={tenant}"

    async def handle(profile: str = Depends(load)) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["profile"] == "tenant=None"


async def test_a_missing_name_without_optional_fails() -> None:
    def load(tenant: str = CallArgument()) -> str: ...

    async def handle(profile: str = Depends(load)) -> None: ...

    async with resolved_dependencies(handle) as deps:
        failure = deps["profile"]

    assert isinstance(failure, FailedDependency)
    assert isinstance(failure.error, LookupError)


async def test_a_top_level_call_argument_references_itself() -> None:
    async def handle(x: str = CallArgument()) -> None: ...

    async with resolved_dependencies(handle) as deps:
        failure = deps["x"]

    assert isinstance(failure, FailedDependency)
    assert isinstance(failure.error, CycleError)
    assert "x -> x" in str(failure.error)


async def test_mutual_references_raise_a_cycle_error() -> None:
    def make_a(b: str = CallArgument("b")) -> str: ...

    def make_b(a: str = CallArgument("a")) -> str: ...

    async def handle(a: str = Depends(make_a), b: str = Depends(make_b)) -> None: ...

    async with resolved_dependencies(handle) as deps:
        first = deps["a"]
        second = deps["b"]

    assert isinstance(first, FailedDependency)
    assert isinstance(first.error, CycleError)
    assert "a -> b -> a" in str(first.error)
    assert isinstance(second, FailedDependency)
    assert second.error is first.error


async def test_optional_does_not_suppress_a_cycle_error() -> None:
    async def handle(x: str = CallArgument(optional=True)) -> None: ...

    async with resolved_dependencies(handle) as deps:
        failure = deps["x"]

    assert isinstance(failure, FailedDependency)
    assert isinstance(failure.error, CycleError)
    assert "x -> x" in str(failure.error)


async def test_a_referenced_sibling_declared_after_is_entered_once() -> None:
    sibling = _Counting()

    def consume(token: str = CallArgument()) -> str:
        return f"used {token}"

    async def handle(
        consumer: str = Depends(consume),
        token: str = cast(str, sibling),
    ) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["consumer"] == "used value-1"
        assert deps["token"] == "value-1"

    assert sibling.enters == 1


async def test_a_referenced_sibling_declared_before_is_entered_once() -> None:
    sibling = _Counting()

    def consume(token: str = CallArgument()) -> str:
        return f"used {token}"

    async def handle(
        token: str = cast(str, sibling),
        consumer: str = Depends(consume),
    ) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["token"] == "value-1"
        assert deps["consumer"] == "used value-1"

    assert sibling.enters == 1


async def test_a_reference_to_a_failed_parameter_shares_the_error() -> None:
    broken = _Exploding()

    def consume(token: str = CallArgument()) -> str: ...

    async def handle(
        token: str = cast(str, broken),
        consumer: str = Depends(consume),
    ) -> None: ...

    async with resolved_dependencies(handle) as deps:
        failed = deps["token"]
        referencing = deps["consumer"]

    assert isinstance(failed, FailedDependency)
    assert isinstance(referencing, FailedDependency)
    assert referencing.error is failed.error
    assert broken.enters == 1


async def test_entering_a_call_argument_without_a_frame_raises() -> None:
    async with AsyncExitStack() as stack:
        _Depends.cache.set({})
        _Depends.stack.set(stack)
        with pytest.raises(RuntimeError, match="frame_scope"):
            await stack.enter_async_context(_CallArgument("value"))


async def test_entering_an_unbound_call_argument_raises() -> None:
    argument = _CallArgument()

    with pytest.raises(RuntimeError, match="never bound to a parameter name"):
        await argument.__aenter__()


def test_for_parameter_returns_a_named_instance_unchanged() -> None:
    argument = _CallArgument("user_id")

    assert argument.for_parameter("account") is argument


def test_for_parameter_copies_a_bare_instance_as_its_own_type() -> None:
    argument = _Quiet(optional=True)

    bound = argument.for_parameter("user_id")

    assert bound is not argument
    assert isinstance(bound, _Quiet)
    assert bound.parameter == "user_id"
    assert bound.optional is True
