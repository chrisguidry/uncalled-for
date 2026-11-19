"""Signature and dependency parameter introspection."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from .annotations import _metadata_dependencies  # pyright: ignore[reportPrivateUsage]
from .base import Dependency, Provided, Provider
from .caching import FunctionCache

_signature_cache: FunctionCache[inspect.Signature] = FunctionCache()


def get_signature(function: Callable[..., object]) -> inspect.Signature:
    """Get a cached signature for a function."""
    cached = _signature_cache.get(function)
    if cached is not None:
        return cached

    signature_attr = getattr(function, "__signature__", None)
    if isinstance(signature_attr, inspect.Signature):
        _signature_cache[function] = signature_attr
        return signature_attr

    signature = inspect.signature(function)
    _signature_cache[function] = signature
    return signature


_parameter_cache: FunctionCache[dict[str, Dependency[object]]] = FunctionCache()


def get_dependency_parameters(
    function: Callable[..., object],
) -> dict[str, Dependency[object]]:
    """Find parameters whose values come from a dependency.

    A parameter qualifies when its default is a ``Dependency``, or when its
    ``Annotated`` metadata holds a ``Provider``, such as ``Depends(...)``. The two
    forms are equivalent, and a parameter may use only one of them.
    """
    cached = _parameter_cache.get(function)
    if cached is not None:
        return cached

    dependencies: dict[str, Dependency[object]] = {}
    signature = get_signature(function)
    metadata = _metadata_dependencies(function)

    for name, parameter in signature.parameters.items():
        providers: list[Dependency[object]] = [
            dependency
            for dependency in metadata.get(name, [])
            if isinstance(dependency, Provider)
        ]
        if parameter.default is Provided and not providers:
            raise TypeError(
                f"Parameter {name!r} of {function.__qualname__} defaults to "
                "Provided, but its annotation has no Provider, such as "
                "Depends(...), to supply the value"
            )
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
