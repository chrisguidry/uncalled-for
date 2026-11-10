"""Call-argument references that add parameters to an outer signature."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

from .base import Dependency
from .frames import _CallArgument  # pyright: ignore[reportPrivateUsage]
from .functional import _Depends  # pyright: ignore[reportPrivateUsage]
from .introspection import get_dependency_parameters, get_signature

_reference_cache: dict[Callable[..., object], list[inspect.Parameter]] = {}


def get_call_argument_parameters(
    function: Callable[..., object],
) -> list[inspect.Parameter]:
    """Find the parameters that dependencies of *function* reference but it lacks.

    A ``CallArgument`` anywhere in the ``Depends`` graph of *function*, in a
    factory's parameters or in a binding, may name a parameter that *function*
    does not declare. Each such name becomes a keyword-only parameter typed
    from the referencing parameter's annotation. It is required unless the
    reference has a *default*. Optional references are left out, so they keep
    yielding ``None``.

    References to one name share a parameter, which is required if any of them
    is. Different annotations or different defaults raise ``TypeError``.
    """
    if function in _reference_cache:
        return _reference_cache[function]

    declared = get_signature(function).parameters
    parameters: dict[str, inspect.Parameter] = {}

    for owner, name, reference in _references(function):
        assert reference.parameter is not None
        if reference.optional or reference.parameter in declared:
            continue

        parameter = inspect.Parameter(
            reference.parameter,
            inspect.Parameter.KEYWORD_ONLY,
            default=reference.default,
            annotation=_annotation(owner, name),
        )
        existing = parameters.get(parameter.name)
        if existing is None:
            parameters[parameter.name] = parameter
            continue
        if existing.annotation != parameter.annotation:
            raise TypeError(
                f"Dependencies of {function.__qualname__} reference "
                f"{parameter.name!r} with different annotations"
            )
        if parameter.default is inspect.Parameter.empty:
            parameters[parameter.name] = parameter
        elif (
            existing.default is not inspect.Parameter.empty
            and existing.default != parameter.default
        ):
            raise TypeError(
                f"Dependencies of {function.__qualname__} reference "
                f"{parameter.name!r} with different defaults"
            )

    result = list(parameters.values())
    _reference_cache[function] = result
    return result


def _references(
    function: Callable[..., Any],
) -> Iterator[tuple[Callable[..., Any], str, _CallArgument]]:
    """Yield each call-argument reference reachable from *function*.

    Each item names the function and parameter that declare the reference,
    which is where its annotation lives.
    """
    for name, dependency in get_dependency_parameters(function).items():
        yield from _dependency_references(function, name, dependency)


def _dependency_references(
    owner: Callable[..., Any],
    name: str,
    dependency: Dependency[Any],
) -> Iterator[tuple[Callable[..., Any], str, _CallArgument]]:
    bound = dependency.for_parameter(name)
    if isinstance(bound, _CallArgument):
        yield owner, name, bound
        return
    if not isinstance(bound, _Depends):
        return

    depends = cast("_Depends[Any]", bound)
    for parameter, inner in get_dependency_parameters(depends.factory).items():
        if parameter not in depends.bindings:
            yield from _dependency_references(depends.factory, parameter, inner)
    for parameter, value in depends.bindings.items():
        if isinstance(value, Dependency):
            yield from _dependency_references(
                depends.factory, parameter, cast(Dependency[Any], value)
            )


def _annotation(owner: Callable[..., Any], name: str) -> Any:
    """The annotation of *owner*'s parameter *name*, without dependency metadata."""
    try:
        hint: Any = get_type_hints(owner, include_extras=True).get(
            name, inspect.Parameter.empty
        )
    except Exception:
        parameter = get_signature(owner).parameters.get(name)
        return parameter.annotation if parameter else inspect.Parameter.empty

    if get_origin(hint) is not Annotated:
        return hint
    base, *metadata = get_args(hint)
    kept = [item for item in metadata if not _is_dependency(item)]
    return Annotated[base, *kept] if kept else base


def _is_dependency(item: object) -> bool:
    return isinstance(item, Dependency) or (
        isinstance(item, type) and issubclass(item, Dependency)
    )
