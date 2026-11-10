"""Tests for call-argument references that add parameters to an outer signature."""

from __future__ import annotations

import inspect
from typing import Annotated, Any

import pytest

from uncalled_for import (
    CallArgument,
    Dependency,
    Depends,
    Shared,
    get_call_argument_parameters,
    without_dependencies,
)


class _Marker(Dependency[None]):
    """An annotation-only dependency that rides along in Annotated metadata."""

    async def __aenter__(self) -> None: ...


def current_workspace(workspace: str | None = CallArgument(default=None)) -> str:
    return workspace or "default"


def current_user(user_id: str = CallArgument()) -> str:
    return f"user:{user_id}"


async def test_an_undeclared_reference_becomes_a_parameter() -> None:
    async def list_items(resolved: str = Depends(current_workspace)) -> str:
        return resolved

    wrapper = without_dependencies(list_items)

    assert str(inspect.signature(wrapper)) == "(*, workspace: str | None = None)"
    assert wrapper.__annotations__ == {"workspace": str | None}
    assert await wrapper(workspace="acme") == "acme"
    assert await wrapper() == "default"


async def test_a_reference_without_a_default_is_required() -> None:
    async def show(name: str, user: str = Depends(current_user)) -> str:
        return f"{name}:{user}"

    wrapper = without_dependencies(show)
    parameter = inspect.signature(wrapper).parameters["user_id"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    assert await wrapper(name="n", user_id="u-1") == "n:user:u-1"


async def test_the_function_never_receives_a_referenced_parameter() -> None:
    received: dict[str, Any] = {}

    def show(user: str = Depends(current_user), **rest: Any) -> str:
        received.update(rest)
        return user

    wrapper = without_dependencies(show)

    assert list(inspect.signature(wrapper).parameters) == ["user_id", "rest"]
    assert await wrapper(user_id="u-1") == "user:u-1"
    assert received == {}


async def test_a_declared_parameter_keeps_its_meaning() -> None:
    async def show(user_id: str, user: str = Depends(current_user)) -> str:
        return f"{user_id}/{user}"

    assert get_call_argument_parameters(show) == []
    assert await without_dependencies(show)(user_id="u-1") == "u-1/user:u-1"


async def test_an_optional_reference_is_not_lifted() -> None:
    def tenant(tenant: str | None = CallArgument(optional=True)) -> str:
        return tenant or "none"

    async def show(value: str = Depends(tenant)) -> str:
        return value

    assert get_call_argument_parameters(show) == []
    assert await without_dependencies(show)() == "none"


async def test_references_through_bindings_and_nested_factories() -> None:
    def account(owner: str, region: str = CallArgument(default="us")) -> str:
        return f"{owner}@{region}"

    def profile(
        acct: str = Depends(account, owner=CallArgument("account_id")),
    ) -> str:
        return acct

    async def show(value: str = Depends(profile)) -> str:
        return value

    parameters = get_call_argument_parameters(show)

    assert [str(p) for p in parameters] == ["region: str = 'us'", "account_id: str"]
    assert await without_dependencies(show)(account_id="a-1") == "a-1@us"


async def test_a_binding_that_overrides_a_reference_hides_it() -> None:
    async def show(user: str = Depends(current_user, user_id="fixed")) -> str: ...

    assert get_call_argument_parameters(show) == []


async def test_shared_dependencies_are_not_walked() -> None:
    def config() -> str: ...

    async def show(value: str = Shared(config)) -> str: ...

    assert get_call_argument_parameters(show) == []


async def test_references_to_one_name_share_a_parameter() -> None:
    def other_workspace(workspace: str | None = CallArgument()) -> str: ...

    async def show(
        a: str = Depends(current_workspace),
        b: str = Depends(other_workspace),
    ) -> str: ...

    async def reversed_show(
        b: str = Depends(other_workspace),
        a: str = Depends(current_workspace),
    ) -> str: ...

    for function in (show, reversed_show):
        (parameter,) = get_call_argument_parameters(function)
        assert parameter.default is inspect.Parameter.empty


async def test_conflicting_annotations_raise() -> None:
    def numeric(workspace: int = CallArgument()) -> int: ...

    async def show(a: str = Depends(current_workspace), b: int = Depends(numeric)): ...

    with pytest.raises(TypeError, match="different annotations"):
        get_call_argument_parameters(show)


async def test_conflicting_defaults_raise() -> None:
    def other(workspace: str | None = CallArgument(default="x")) -> str: ...

    def same(workspace: str | None = CallArgument(default=None)) -> str: ...

    async def show(a: str = Depends(current_workspace), b: str = Depends(other)): ...

    async def agreeing(a: str = Depends(current_workspace), b: str = Depends(same)): ...

    with pytest.raises(TypeError, match="different defaults"):
        get_call_argument_parameters(show)
    (parameter,) = get_call_argument_parameters(agreeing)
    assert parameter.default is None


async def test_annotated_references_drop_their_dependency_metadata() -> None:
    def tagged(
        workspace: Annotated[str, "described", CallArgument(default="w")],
    ) -> str: ...

    def bare(workspace: Annotated[str, _Marker, CallArgument()]) -> str: ...

    async def show(a: str = Depends(tagged)) -> str: ...

    async def plain(a: str = Depends(bare)) -> str: ...

    (tagged_parameter,) = get_call_argument_parameters(show)
    (bare_parameter,) = get_call_argument_parameters(plain)

    assert tagged_parameter.annotation == Annotated[str, "described"]
    assert bare_parameter.annotation is str


async def test_unresolvable_hints_fall_back_to_the_raw_annotation() -> None:
    def broken(
        workspace: "UnresolvableType" = CallArgument(),  # pyright: ignore[reportUndefinedVariable,reportUnknownParameterType]  # noqa: F821
    ) -> str: ...

    async def show(a: str = Depends(broken)) -> str:  # pyright: ignore[reportUnknownArgumentType]
        ...

    (parameter,) = get_call_argument_parameters(show)
    assert parameter.annotation == "'UnresolvableType'"


def test_optional_and_default_are_exclusive() -> None:
    with pytest.raises(TypeError, match="optional or default"):
        CallArgument(optional=True, default=None)
