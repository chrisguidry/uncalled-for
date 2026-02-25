"""Tests for signature and parameter introspection."""

from __future__ import annotations

import inspect
from typing import cast

from uncalled_for import Dependency, Depends, get_dependency_parameters, get_signature


class _SimpleDep(Dependency[str]):
    async def __aenter__(self) -> str: ...


def SimpleDep() -> str:
    return cast(str, _SimpleDep())


def test_get_signature_caches() -> None:
    def my_func(x: int) -> None: ...

    sig1 = get_signature(my_func)
    sig2 = get_signature(my_func)
    assert sig1 is sig2


def test_get_signature_respects_dunder_signature() -> None:
    def my_func() -> None: ...

    custom_sig = inspect.Signature(
        parameters=[inspect.Parameter("y", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    my_func.__signature__ = custom_sig  # pyright: ignore[reportFunctionMemberAccess]

    sig = get_signature(my_func)
    assert "y" in sig.parameters


def test_get_dependency_parameters_finds_dependencies() -> None:
    def get_value() -> str: ...

    async def my_func(
        regular: str,
        dep: str = SimpleDep(),
        functional: str = Depends(get_value),
        default: int = 5,
    ) -> None: ...

    params = get_dependency_parameters(my_func)
    assert set(params.keys()) == {"dep", "functional"}


def test_get_dependency_parameters_caches() -> None:
    async def my_func(dep: str = SimpleDep()) -> None: ...

    params1 = get_dependency_parameters(my_func)
    params2 = get_dependency_parameters(my_func)
    assert params1 is params2
