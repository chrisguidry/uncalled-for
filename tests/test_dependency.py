"""Tests for the Dependency base class."""

from __future__ import annotations

import pytest

from uncalled_for import Dependency


class Greeter(Dependency[str]):
    async def __aenter__(self) -> str:
        return "hello"


class Farewell(Dependency[str]):
    async def __aenter__(self) -> str:
        return "goodbye"

    async def __aexit__(self, *args: object) -> None:
        pass


def test_dependency_is_abstract() -> None:
    with pytest.raises(TypeError):
        Dependency()  # pyright: ignore[reportAbstractUsage]


def test_single_defaults_to_false() -> None:
    assert Greeter.single is False


async def test_aenter_produces_value() -> None:
    async with Greeter() as value:
        assert value == "hello"


async def test_aexit_runs_without_error() -> None:
    async with Farewell() as value:
        assert value == "goodbye"


def test_single_flag() -> None:
    class SingletonDep(Dependency[str]):
        single = True

        async def __aenter__(self) -> str: ...

    assert SingletonDep.single is True
