"""Signature and dependency parameter introspection."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from .annotations import get_annotated_metadata
from .base import Dependency

_signature_cache: dict[Callable[..., Any], inspect.Signature] = {}


def get_signature(function: Callable[..., Any]) -> inspect.Signature:
    """Get a cached signature for a function."""
    if function in _signature_cache:
        return _signature_cache[function]

    signature_attr = getattr(function, "__signature__", None)
    if isinstance(signature_attr, inspect.Signature):
        _signature_cache[function] = signature_attr
        return signature_attr

    signature = inspect.signature(function)
    _signature_cache[function] = signature
    return signature


_parameter_cache: dict[Callable[..., Any], dict[str, Dependency[Any]]] = {}


def get_dependency_parameters(
    function: Callable[..., Any],
) -> dict[str, Dependency[Any]]:
    """Find parameters whose values come from a dependency.

    A parameter qualifies when its default is a ``Dependency``, or when its
    ``Annotated`` metadata holds a ``Depends(...)``, ``Shared(...)`` or
    ``CallArgument(...)``. The two
    forms are equivalent, and a parameter may use only one of them.
    """
    if function in _parameter_cache:
        return _parameter_cache[function]

    dependencies: dict[str, Dependency[Any]] = {}
    signature = get_signature(function)
    metadata = get_annotated_metadata(function)

    for name, parameter in signature.parameters.items():
        providers: list[Dependency[Any]] = [
            dependency
            for dependency in metadata.get(name, [])
            if dependency._annotated_provides_value  # pyright: ignore[reportPrivateUsage]
        ]
        if isinstance(parameter.default, Dependency):
            providers.append(parameter.default)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        if len(providers) > 1:
            raise TypeError(
                f"Parameter {name!r} of {function.__qualname__} declares more "
                "than one dependency for its value"
            )
        if providers:
            dependencies[name] = providers[0]

    _parameter_cache[function] = dependencies
    return dependencies
