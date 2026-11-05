"""Tests for Depends, Shared and CallArgument declared as Annotated metadata."""

import inspect
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated

import pytest

from uncalled_for import (
    CallArgument,
    CycleError,
    Dependency,
    Depends,
    FailedDependency,
    Provider,
    Shared,
    SharedContext,
    get_annotation_dependencies,
    get_dependency_parameters,
    resolved_dependencies,
    validate_dependencies,
    without_dependencies,
)


def get_db() -> str:
    return "db"


DB = Annotated[str, Depends(get_db)]


def test_annotated_depends_is_a_dependency_parameter() -> None:
    async def get_user(email: str, db: DB) -> None: ...

    parameters = get_dependency_parameters(get_user)
    assert set(parameters) == {"db"}


def test_annotated_depends_is_not_an_annotation_dependency() -> None:
    async def get_user(email: str, db: DB) -> None: ...

    assert get_annotation_dependencies(get_user) == {}


def test_annotated_and_default_forms_are_found_together() -> None:
    async def handler(
        a: DB,
        b: str = Depends(get_db),
        c: int = 5,
    ) -> None: ...

    assert set(get_dependency_parameters(handler)) == {"a", "b"}


async def test_annotated_depends_resolves_value() -> None:
    async def get_user(email: str, db: DB) -> None: ...

    async with resolved_dependencies(get_user, {"email": "x"}) as arguments:
        assert arguments == {"db": "db"}


async def test_caller_value_overrides_annotated_depends() -> None:
    async def handler(db: DB) -> None: ...

    async with resolved_dependencies(handler, {"db": "mine"}) as arguments:
        assert arguments == {"db": "mine"}


async def test_annotated_depends_generator_is_cleaned_up() -> None:
    events: list[str] = []

    @asynccontextmanager
    async def session() -> AsyncGenerator[str]:
        events.append("open")
        yield "session"
        events.append("close")

    async def handler(db: Annotated[str, Depends(session)]) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments["db"] == "session"
        assert events == ["open"]
    assert events == ["open", "close"]


async def test_annotated_depends_resolves_once_per_call() -> None:
    calls = 0

    def factory() -> str:
        nonlocal calls
        calls += 1
        return "value"

    async def handler(db: Annotated[str, Depends(factory)]) -> None: ...

    async with resolved_dependencies(handler):
        pass
    assert calls == 1


async def test_annotated_depends_in_a_factory() -> None:
    def get_account(db: DB) -> str:
        return f"account from {db}"

    async def handler(
        account: Annotated[str, Depends(get_account)],
    ) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments["account"] == "account from db"


async def test_annotated_depends_with_call_argument_binding() -> None:
    def load_user(user_id: str) -> str:
        return f"user {user_id}"

    async def handler(
        user_id: str,
        user: Annotated[str, Depends(load_user, user_id=CallArgument("user_id"))],
    ) -> None: ...

    async with resolved_dependencies(handler, {"user_id": "7"}) as arguments:
        assert arguments["user"] == "user 7"


async def test_call_argument_sees_annotated_depends_value() -> None:
    def describe(db: str = CallArgument()) -> str:
        return f"described {db}"

    async def handler(
        db: DB,
        description: str = Depends(describe),
    ) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments["description"] == "described db"


async def test_annotated_shared_resolves_once_across_calls() -> None:
    calls = 0

    async def pool() -> str:
        nonlocal calls
        calls += 1
        return "pool"

    async def handler(p: Annotated[str, Shared(pool)]) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(handler) as first:
            assert first["p"] == "pool"
        async with resolved_dependencies(handler) as second:
            assert second["p"] == "pool"
    assert calls == 1


async def test_side_effect_metadata_sees_annotated_depends_value() -> None:
    bound: list[tuple[str, object]] = []

    class Recorder(Dependency["Recorder"]):
        def bind_to_parameter(self, name: str, value: object) -> "Recorder":
            bound.append((name, value))
            return self

        async def __aenter__(self) -> "Recorder":
            return self

    async def handler(db: Annotated[str, Depends(get_db), Recorder()]) -> None: ...

    assert set(get_dependency_parameters(handler)) == {"db"}
    assert [type(d) for d in get_annotation_dependencies(handler)["db"]] == [Recorder]

    async with resolved_dependencies(handler) as arguments:
        assert arguments["db"] == "db"
    assert bound == [("db", "db")]


def test_two_annotated_value_dependencies_raise() -> None:
    def other() -> str:
        return "other"  # pragma: no cover

    async def handler(db: Annotated[str, Depends(get_db), Depends(other)]) -> None: ...

    with pytest.raises(TypeError, match="more than one dependency"):
        get_dependency_parameters(handler)


def test_annotated_depends_with_dependency_default_raises() -> None:
    async def handler(db: DB = Depends(get_db)) -> None: ...

    with pytest.raises(TypeError, match="more than one dependency"):
        get_dependency_parameters(handler)


async def test_without_dependencies_hides_annotated_depends() -> None:
    async def get_user(email: str, db: DB) -> str:
        return f"{email} via {db}"

    wrapper = without_dependencies(get_user)
    signature = inspect.signature(wrapper)

    assert list(signature.parameters) == ["email"]
    assert "db" not in wrapper.__annotations__
    assert await wrapper(email="foo@example.com") == "foo@example.com via db"


async def test_annotated_sync_context_manager_is_cleaned_up() -> None:
    events: list[str] = []

    @contextmanager
    def session() -> Generator[str]:
        events.append("open")
        yield "session"
        events.append("close")

    async def handler(db: Annotated[str, Depends(session)]) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments["db"] == "session"
        assert events == ["open"]
    assert events == ["open", "close"]


async def test_annotated_async_factory() -> None:
    async def factory() -> str:
        return "async"

    async def handler(value: Annotated[str, Depends(factory)]) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments["value"] == "async"


async def test_without_dependencies_with_sync_function() -> None:
    def get_user(email: str, db: DB) -> str:
        return f"{email} via {db}"

    assert await without_dependencies(get_user)(email="a") == "a via db"


async def test_one_cached_value_across_annotated_default_and_nested() -> None:
    calls = 0

    def counted() -> str:
        nonlocal calls
        calls += 1
        return "c"

    def inner(c: Annotated[str, Depends(counted)]) -> str:
        return f"inner {c}"

    async def handler(
        a: Annotated[str, Depends(counted)],
        i: Annotated[str, Depends(inner)],
        b: str = Depends(counted),
    ) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments == {"a": "c", "i": "inner c", "b": "c"}
    assert calls == 1


async def test_matching_bindings_share_across_forms() -> None:
    calls = 0

    def factory(key: str) -> str:
        nonlocal calls
        calls += 1
        return key

    async def handler(
        p: Annotated[str, Depends(factory, key="z")],
        q: str = Depends(factory, key="z"),
    ) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments == {"p": "z", "q": "z"}
    assert calls == 1


async def test_failing_annotated_depends_becomes_failed_dependency() -> None:
    def boom() -> str:
        raise RuntimeError("boom")

    async def handler(db: Annotated[str, Depends(boom)]) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        failure = arguments["db"]
        assert isinstance(failure, FailedDependency)
        assert str(failure.error) == "boom"


async def test_annotated_call_argument_cycle_is_reported() -> None:
    def from_y(value: str = CallArgument("y")) -> str: ...

    def from_x(value: str = CallArgument("x")) -> str: ...

    async def handler(
        x: Annotated[str, Depends(from_y)],
        y: Annotated[str, Depends(from_x)],
    ) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        failure = arguments["x"]
        assert isinstance(failure, FailedDependency)
        assert isinstance(failure.error, CycleError)


async def test_annotated_call_argument_in_a_factory() -> None:
    def load_user(user_id: Annotated[str, CallArgument()]) -> str:
        return f"user {user_id}"

    async def handler(
        user_id: str,
        user: Annotated[str, Depends(load_user)],
    ) -> None: ...

    async with resolved_dependencies(handler, {"user_id": "7"}) as arguments:
        assert arguments["user"] == "user 7"


async def test_annotated_named_call_argument_in_a_factory() -> None:
    def load_user(uid: Annotated[str, CallArgument("owner")]) -> str:
        return f"user {uid}"

    async def handler(
        owner: str,
        user: Annotated[str, Depends(load_user)],
    ) -> None: ...

    async with resolved_dependencies(handler, {"owner": "9"}) as arguments:
        assert arguments["user"] == "user 9"


def test_validate_counts_annotated_depends_once() -> None:
    """A value dependency is reported once, not as both kinds of dependency."""

    class Once(Provider[str]):
        single = True

        async def __aenter__(self) -> str: ...

    async def handler(x: Annotated[str, Once()]) -> None: ...

    validate_dependencies(handler)


def test_unresolvable_hints_still_find_default_dependencies() -> None:
    async def handler(
        x: "UnresolvableType",  # pyright: ignore[reportUndefinedVariable,reportUnknownParameterType]  # noqa: F821
        db: str = Depends(get_db),
    ) -> None: ...

    assert set(get_dependency_parameters(handler)) == {"db"}  # pyright: ignore[reportUnknownArgumentType]


def test_get_dependency_parameters_caches_annotated_dependencies() -> None:
    async def handler(db: DB) -> None: ...

    assert get_dependency_parameters(handler) is get_dependency_parameters(handler)


def test_bare_class_with_required_args_raises_from_dependency_parameters() -> None:
    class NeedsArgs(Dependency[str]):
        def __init__(self, required: int) -> None: ...

        async def __aenter__(self) -> str: ...

    async def handler(x: Annotated[int, NeedsArgs]) -> None: ...

    with pytest.raises(TypeError):
        get_dependency_parameters(handler)


def test_annotated_shared_with_dependency_default_raises() -> None:
    async def pool() -> str: ...

    async def handler(p: Annotated[str, Shared(pool)] = Depends(get_db)) -> None: ...

    with pytest.raises(TypeError, match="more than one dependency"):
        get_dependency_parameters(handler)


async def test_annotated_depends_ignores_a_plain_default() -> None:
    async def handler(db: Annotated[str, Depends(get_db)] = "fallback") -> None: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments["db"] == "db"


async def test_annotated_depends_on_a_keyword_only_parameter() -> None:
    async def handler(email: str, *, db: DB) -> str:
        return f"{email} via {db}"

    assert await without_dependencies(handler)(email="a") == "a via db"


async def test_annotated_depends_in_a_shared_factory() -> None:
    async def config() -> str:
        return "config"

    async def pool(c: Annotated[str, Shared(config)]) -> str:
        return f"pool from {c}"

    async def handler(p: Annotated[str, Shared(pool)]) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(handler) as arguments:
            assert arguments["p"] == "pool from config"


async def test_binding_replaces_an_annotated_factory_parameter() -> None:
    def never() -> str:
        raise AssertionError("never resolved")  # pragma: no cover

    def load(key: Annotated[str, Depends(never)]) -> str:
        return f"loaded {key}"

    async def handler(value: str = Depends(load, key="bound")) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments["value"] == "loaded bound"


async def test_annotated_optional_call_argument_yields_none() -> None:
    def load(tenant: Annotated[str | None, CallArgument(optional=True)]) -> str:
        return f"tenant {tenant}"

    async def handler(value: Annotated[str, Depends(load)]) -> None: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments["value"] == "tenant None"


async def test_caller_value_overrides_annotated_depends_in_wrapper() -> None:
    async def get_user(email: str, db: DB) -> str:
        return f"{email} via {db}"

    wrapper = without_dependencies(get_user)
    assert await wrapper(email="a", db="mine") == "a via mine"


async def test_custom_provider_supplies_value_as_metadata() -> None:
    class Current(Provider[str]):
        async def __aenter__(self) -> str:
            return "current"

    async def handler(value: Annotated[str, Current()]) -> None: ...

    assert set(get_dependency_parameters(handler)) == {"value"}
    assert get_annotation_dependencies(handler) == {}

    async with resolved_dependencies(handler) as arguments:
        assert arguments == {"value": "current"}


async def test_custom_dependency_wraps_the_call_as_metadata() -> None:
    entered: list[object] = []

    class Wrapper(Dependency["Wrapper"]):
        def bind_to_parameter(self, name: str, value: object) -> "Wrapper":
            entered.append(value)
            return self

        async def __aenter__(self) -> "Wrapper":
            return self

    async def handler(value: Annotated[int, Wrapper()]) -> None: ...

    assert get_dependency_parameters(handler) == {}

    async with resolved_dependencies(handler, {"value": 3}) as arguments:
        assert arguments == {}
    assert entered == [3]


def test_bare_provider_class_supplies_value_as_metadata() -> None:
    class Current(Provider[str]):
        async def __aenter__(self) -> str: ...

    async def handler(value: Annotated[str, Current]) -> None: ...

    assert isinstance(get_dependency_parameters(handler)["value"], Current)


def test_builtin_dependencies_are_providers() -> None:
    def factory() -> str: ...

    assert isinstance(Depends(factory), Provider)
    assert isinstance(Shared(factory), Provider)
    assert isinstance(CallArgument(), Provider)
