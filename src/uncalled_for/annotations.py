"""Annotation-based dependency extraction from ``Annotated`` type hints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, cast, get_args, get_origin, get_type_hints

from .base import Dependency, Provider

_metadata_cache: dict[Callable[..., object], dict[str, list[Dependency[object]]]] = {}
_annotation_cache: dict[Callable[..., object], dict[str, list[Dependency[object]]]] = {}


def get_annotated_metadata(
    function: Callable[..., object],
) -> dict[str, list[Dependency[object]]]:
    """Find every ``Dependency`` in the ``Annotated`` metadata of a function."""
    if function in _metadata_cache:
        return _metadata_cache[function]

    result: dict[str, list[Dependency[object]]] = {}
    try:
        hints = get_type_hints(function, include_extras=True)
    except Exception:
        _metadata_cache[function] = result
        return result

    for name, hint in hints.items():
        if name == "return":
            continue
        if get_origin(hint) is not Annotated:
            continue
        dependencies: list[Dependency[object]] = []
        for a in get_args(hint)[1:]:
            if isinstance(a, Dependency):
                dependencies.append(cast(Dependency[object], a))
            elif isinstance(a, type) and issubclass(a, Dependency):
                dependencies.append(cast(Dependency[object], a()))
        if dependencies:
            result[name] = dependencies

    _metadata_cache[function] = result
    return result


def get_annotation_dependencies(
    function: Callable[..., object],
) -> dict[str, list[Dependency[object]]]:
    """Find ``Dependency`` instances in ``Annotated`` type-hint metadata.

    Bare ``Dependency`` subclasses are also accepted as a shorthand for a
    parameterless instance: ``Annotated[T, Dep]`` is equivalent to
    ``Annotated[T, Dep()]``.

    These dependencies wrap the call but do not supply the parameter's value.
    ``Provider`` metadata, such as ``Depends(...)``, does supply the value, so
    it is reported by ``get_dependency_parameters`` instead.
    """
    if function in _annotation_cache:
        return _annotation_cache[function]

    result: dict[str, list[Dependency[object]]] = {}
    for name, dependencies in get_annotated_metadata(function).items():
        side_effects = [d for d in dependencies if not isinstance(d, Provider)]
        if side_effects:
            result[name] = side_effects

    _annotation_cache[function] = result
    return result
