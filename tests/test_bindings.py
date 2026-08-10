"""Tests for keyword bindings on Depends()."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AsyncExitStack, contextmanager
from typing import Any, cast

from uncalled_for import (
    CallArgument,
    Dependency,
    Depends,
    FailedDependency,
    resolved_dependencies,
)
from uncalled_for.functional import _Depends  # pyright: ignore[reportPrivateUsage]


class _Token(Dependency[str]):
    """Produces a value, counts its entries, and compares by that value."""

    def __init__(self, value: str) -> None:
        self.value = value
        self.enters = 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Token) and other.value == self.value

    async def __aenter__(self) -> str:
        self.enters += 1
        return self.value


async def test_a_plain_binding_reaches_the_factory() -> None:
    def make(x: str) -> str:
        return f"made {x}"

    async def handle(value: str = Depends(make, x="here")) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["value"] == "made here"


async def test_a_binding_replaces_a_plain_default() -> None:
    def make(x: str = "default") -> str:
        return f"made {x}"

    async def handle(value: str = Depends(make, x="bound")) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["value"] == "made bound"


async def test_a_binding_replaces_a_dependency_default() -> None:
    token = _Token("real")

    def get_db() -> str: ...

    def query(db: str = Depends(get_db), user: str = cast(str, token)) -> str:
        return f"{db} for {user}"

    async def handle(result: str = Depends(query, db="fake", user="guest")) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["result"] == "fake for guest"

    assert token.enters == 0


async def test_a_binding_joins_the_factorys_own_dependencies() -> None:
    def get_db() -> str:
        return "db"

    def query(db: str = Depends(get_db), table: str = "default") -> str:
        return f"{db}:{table}"

    async def handle(result: str = Depends(query, table="users")) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["result"] == "db:users"


async def test_a_context_manager_binding_wraps_the_factory() -> None:
    events: list[str] = []

    @contextmanager
    def open_handle() -> Iterator[str]:
        events.append("open")
        yield "handle"
        events.append("close")

    def read(source: str) -> str:
        events.append(f"read {source}")
        return "contents"

    async def load(text: str = Depends(read, source=Depends(open_handle))) -> None: ...

    async with resolved_dependencies(load) as deps:
        assert deps["text"] == "contents"
        assert events == ["open", "read handle"]

    assert events == ["open", "read handle", "close"]


async def test_a_bare_call_argument_binding_uses_the_factory_parameter_name() -> None:
    def make(x: str) -> str:
        return f"made {x}"

    async def handle(x: str, value: str = Depends(make, x=CallArgument())) -> None: ...

    async with resolved_dependencies(handle, {"x": "here"}) as deps:
        assert deps["value"] == "made here"


async def test_a_named_call_argument_binding_maps_a_different_name() -> None:
    def make(x: str) -> str:
        return f"made {x}"

    async def handle(
        outer_name: str,
        value: str = Depends(make, x=CallArgument("outer_name")),
    ) -> None: ...

    async with resolved_dependencies(handle, {"outer_name": "there"}) as deps:
        assert deps["value"] == "made there"


async def test_matching_call_argument_bindings_share_a_cache_entry() -> None:
    calls = 0

    def make(x: str) -> str:
        nonlocal calls
        calls += 1
        return f"made {x}"

    async def handle(
        a: str,
        first: str = Depends(make, x=CallArgument("a")),
        second: str = Depends(make, x=CallArgument("a")),
    ) -> None: ...

    async with resolved_dependencies(handle, {"a": "one"}) as deps:
        assert deps["first"] == "made one"
        assert deps["second"] == "made one"

    assert calls == 1


async def test_bare_and_explicit_call_argument_bindings_share_a_cache_entry() -> None:
    calls = 0

    def make(x: str) -> str:
        nonlocal calls
        calls += 1
        return f"made {x}"

    async def handle(
        x: str,
        first: str = Depends(make, x=CallArgument()),
        second: str = Depends(make, x=CallArgument("x")),
    ) -> None: ...

    async with resolved_dependencies(handle, {"x": "one"}) as deps:
        assert deps["first"] == "made one"
        assert deps["second"] == "made one"

    assert calls == 1


async def test_differing_call_argument_bindings_resolve_separately() -> None:
    seen: list[str] = []

    def make(x: str) -> str:
        seen.append(x)
        return f"made {x}"

    async def handle(
        a: str,
        b: str,
        first: str = Depends(make, x=CallArgument("a")),
        second: str = Depends(make, x=CallArgument("b")),
    ) -> None: ...

    async with resolved_dependencies(handle, {"a": "one", "b": "two"}) as deps:
        assert deps["first"] == "made one"
        assert deps["second"] == "made two"

    assert seen == ["one", "two"]


async def test_bare_depends_cache_key_is_the_factory() -> None:
    def get_value() -> str:
        return "cached"

    cache: dict[Any, Any] = {}

    async with AsyncExitStack() as stack:
        _Depends.cache.set(cache)
        _Depends.stack.set(stack)
        assert await stack.enter_async_context(_Depends(get_value)) == "cached"

    assert cache == {get_value: "cached"}


async def test_matching_plain_bindings_share_a_cache_entry() -> None:
    calls = 0

    def make(x: int) -> str:
        nonlocal calls
        calls += 1
        return f"made {x}"

    async def handle(
        first: str = Depends(make, x=42),
        second: str = Depends(make, x=42),
    ) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["first"] == "made 42"
        assert deps["second"] == "made 42"

    assert calls == 1


async def test_differing_plain_bindings_resolve_separately() -> None:
    calls = 0

    def make(x: int) -> str:
        nonlocal calls
        calls += 1
        return f"made {x}"

    async def handle(
        first: str = Depends(make, x=42),
        second: str = Depends(make, x=43),
    ) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["first"] == "made 42"
        assert deps["second"] == "made 43"

    assert calls == 2


async def test_equal_values_of_different_types_resolve_separately() -> None:
    seen: list[str] = []

    def make(x: object) -> str:
        seen.append(f"{type(x).__name__}:{x}")
        return f"made {x}"

    async def handle(
        first: str = Depends(make, x=1),
        second: str = Depends(make, x=1.0),
        third: str = Depends(make, x=True),
    ) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["first"] == "made 1"
        assert deps["second"] == "made 1.0"
        assert deps["third"] == "made True"

    assert seen == ["int:1", "float:1.0", "bool:True"]


async def test_unhashable_bindings_resolve_separately() -> None:
    seen: list[list[int]] = []

    def make(x: list[int]) -> str:
        seen.append(x)
        return f"made {x}"

    async def handle(
        first: str = Depends(make, x=[1, 2]),
        second: str = Depends(make, x=[1, 2]),
    ) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["first"] == "made [1, 2]"
        assert deps["second"] == "made [1, 2]"

    assert seen == [[1, 2], [1, 2]]


async def test_the_same_dependency_binding_shares_a_cache_entry() -> None:
    calls = 0
    token = _Token("t")

    def make(x: str) -> str:
        nonlocal calls
        calls += 1
        return f"made {x}"

    async def handle(
        first: str = Depends(make, x=token),
        second: str = Depends(make, x=token),
    ) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["first"] == "made t"
        assert deps["second"] == "made t"

    assert calls == 1
    assert token.enters == 1


async def test_equal_dependency_bindings_resolve_separately() -> None:
    calls = 0
    first_token = _Token("t")
    second_token = _Token("t")

    assert first_token == second_token

    def make(x: str) -> str:
        nonlocal calls
        calls += 1
        return f"made {x}"

    async def handle(
        first: str = Depends(make, x=first_token),
        second: str = Depends(make, x=second_token),
    ) -> None: ...

    async with resolved_dependencies(handle) as deps:
        assert deps["first"] == "made t"
        assert deps["second"] == "made t"

    assert calls == 2
    assert first_token.enters == 1
    assert second_token.enters == 1


async def test_nested_bindings_share_a_cache_entry() -> None:
    outer_calls = 0
    inner_calls = 0

    def inner(y: str) -> str:
        nonlocal inner_calls
        inner_calls += 1
        return f"inner {y}"

    def outer(x: str) -> str:
        nonlocal outer_calls
        outer_calls += 1
        return f"outer {x}"

    async def handle(
        a: str,
        first: str = Depends(outer, x=Depends(inner, y=CallArgument("a"))),
        second: str = Depends(outer, x=Depends(inner, y=CallArgument("a"))),
    ) -> None: ...

    async with resolved_dependencies(handle, {"a": "seed"}) as deps:
        assert deps["first"] == "outer inner seed"
        assert deps["second"] == "outer inner seed"

    assert outer_calls == 1
    assert inner_calls == 1


async def test_an_unknown_binding_name_fails_the_parameter() -> None:
    def make(x: str) -> str: ...

    async def handle(value: str = Depends(make, y="oops")) -> None: ...

    async with resolved_dependencies(handle) as deps:
        failure = deps["value"]

    assert isinstance(failure, FailedDependency)
    assert isinstance(failure.error, TypeError)


async def test_bindings_resolve_without_a_frame() -> None:
    def get_extra() -> str:
        return "extra"

    def combine(x: str, y: str) -> str:
        return f"{x}+{y}"

    async with AsyncExitStack() as stack:
        _Depends.cache.set({})
        _Depends.stack.set(stack)
        value = await stack.enter_async_context(
            _Depends(combine, x="plain", y=Depends(get_extra))
        )

    assert value == "plain+extra"
