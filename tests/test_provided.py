"""Tests for the Provided default on parameters whose value comes from metadata."""

import copy
import inspect
import pickle
from collections.abc import Callable
from typing import Annotated

import pytest

from uncalled_for import (
    Dependency,
    Depends,
    Provided,
    get_dependency_parameters,
    resolved_dependencies,
    without_dependencies,
)


def get_db() -> str:
    return "db"


DB = Annotated[str, Depends(get_db)]


def test_provided_repr() -> None:
    assert repr(Provided) == "Provided"


def round_trip_through_pickle(value: object) -> object:
    return pickle.loads(pickle.dumps(value))


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(round_trip_through_pickle, id="pickle"),
    ],
)
def test_provided_keeps_its_identity(duplicate: Callable[[object], object]) -> None:
    assert duplicate(Provided) is Provided


async def test_provided_parameter_resolves_from_its_annotation() -> None:
    async def get_user(email: str, db: DB = Provided) -> None: ...

    assert set(get_dependency_parameters(get_user)) == {"db"}
    async with resolved_dependencies(get_user, {"email": "a"}) as arguments:
        assert arguments == {"db": "db"}


async def test_caller_value_overrides_provided() -> None:
    async def get_user(db: DB = Provided) -> None: ...

    async with resolved_dependencies(get_user, {"db": "mine"}) as arguments:
        assert arguments == {"db": "mine"}


async def test_without_dependencies_hides_a_provided_parameter() -> None:
    async def get_user(email: str, db: DB = Provided) -> str:
        return f"{email} via {db}"

    wrapper = without_dependencies(get_user)

    assert list(inspect.signature(wrapper).parameters) == ["email"]
    assert await wrapper(email="a") == "a via db"


class Wrapper(Dependency["Wrapper"]):
    async def __aenter__(self) -> "Wrapper": ...


async def no_annotation(db: str = Provided) -> None: ...


async def no_metadata(db: Annotated[str, "not a dependency"] = Provided) -> None: ...


async def only_a_wrapper(db: Annotated[str, Wrapper()] = Provided) -> None: ...


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(no_annotation, id="no-annotation"),
        pytest.param(no_metadata, id="no-dependency-metadata"),
        pytest.param(only_a_wrapper, id="dependency-that-is-not-a-provider"),
    ],
)
def test_provided_without_a_provider_raises(
    function: Callable[..., object],
) -> None:
    with pytest.raises(TypeError, match="'db' .* defaults to Provided"):
        get_dependency_parameters(function)
