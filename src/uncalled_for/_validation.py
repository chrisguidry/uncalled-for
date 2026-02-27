"""Dependency declaration validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from ._annotations import get_annotation_dependencies
from ._base import Dependency
from ._introspection import get_dependency_parameters


def validate_dependencies(function: Callable[..., Any]) -> None:
    """Check that a function's dependency declarations are valid.

    Raises ``ValueError`` if multiple dependencies with ``single=True``
    share the same type or base class.

    Concrete-type duplicates are checked first so the error message names
    the exact type (e.g. "Retry") rather than an abstract ancestor
    (e.g. "FailureHandler").
    """
    parameters = get_dependency_parameters(function)
    dependencies = list(parameters.values())

    counts: Counter[type[Dependency[Any]]] = Counter(
        type(dependency)
        for dependency in dependencies  # pyright: ignore[reportUnknownArgumentType]
    )
    for dependency_type, count in counts.items():
        if getattr(dependency_type, "single", False) and count > 1:  # pyright: ignore[reportUnknownArgumentType]
            raise ValueError(
                f"Only one {dependency_type.__name__} dependency is allowed"  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
            )

    single_bases: set[type[Dependency[Any]]] = set()
    for dependency in dependencies:
        for cls in type(dependency).__mro__:
            if (
                issubclass(cls, Dependency)
                and cls is not Dependency
                and getattr(cls, "single", False)  # pyright: ignore[reportUnknownArgumentType]
            ):
                single_bases.add(cls)  # pyright: ignore[reportUnknownArgumentType]

    for base_class in single_bases:
        instances = [
            dependency
            for dependency in dependencies
            if isinstance(dependency, base_class)
        ]
        if len(instances) > 1:
            types = ", ".join(type(instance).__name__ for instance in instances)
            raise ValueError(
                f"Only one {base_class.__name__} dependency is allowed, "
                f"but found: {types}"
            )

    annotation_dependencies = get_annotation_dependencies(function)
    for parameter_name, parameter_dependencies in annotation_dependencies.items():
        annotation_counts: Counter[type[Dependency[Any]]] = Counter(
            type(dependency)
            for dependency in parameter_dependencies  # pyright: ignore[reportUnknownArgumentType]
        )
        for dependency_type, count in annotation_counts.items():
            if getattr(dependency_type, "single", False) and count > 1:  # pyright: ignore[reportUnknownArgumentType]
                raise ValueError(
                    f"Only one {dependency_type.__name__} annotation dependency "  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                    f"is allowed per parameter, but found {count} on '{parameter_name}'"
                )
