"""Base dependency class."""

from __future__ import annotations

import abc
from types import TracebackType
from typing import Any, Generic, Never, TypeVar, cast

T = TypeVar("T", covariant=True)


class Dependency(abc.ABC, Generic[T]):
    """Base class for all injectable dependencies.

    Subclasses implement ``__aenter__`` to produce the injected value and
    optionally ``__aexit__`` for cleanup. The resolution engine enters each
    dependency as an async context manager, so resources are cleaned up in
    reverse order when the call completes.

    Set ``single = True`` on a subclass to enforce that only one instance
    of that dependency type may appear in a function's signature.
    """

    single: bool = False

    def bind_to_parameter(self, name: str, value: Any) -> Dependency[T]:
        """Return a copy bound to a parameter's name and value.

        Called when the dependency appears as ``Annotated`` metadata.
        Subclasses override to capture context; the default returns *self*.
        """
        return self

    def for_parameter(self, name: str) -> Dependency[T]:
        """Return the dependency to use for the parameter called *name*.

        Called with the name of the parameter this dependency is about to
        resolve for. The default returns *self*. Subclasses override to
        return a copy bound to the name.

        This is not ``bind_to_parameter``, which applies to ``Annotated``
        metadata and also receives the parameter's value.
        """
        return self

    @abc.abstractmethod
    async def __aenter__(self) -> T: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass


class Provider(Dependency[T]):
    """A dependency that supplies a parameter's value wherever it is declared.

    As a default, any ``Dependency`` supplies the value. As ``Annotated``
    metadata, only a ``Provider`` does: ``Annotated[T, Depends(factory)]``
    resolves like a ``Depends(factory)`` default. Other dependencies in
    metadata wrap the call and receive the parameter's value through
    ``bind_to_parameter``.

    ``Depends``, ``Shared`` and ``CallArgument`` are providers. Subclass
    ``Provider`` instead of ``Dependency`` for a custom dependency that
    should supply the value when used as metadata.
    """


class _Provided:
    """The type of ``Provided``, which has exactly one instance."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "Provided"

    def __reduce__(self) -> str:
        # Copies and unpickled values must still be ``Provided``, because the
        # resolution engine checks for it by identity.
        return "Provided"


Provided: Never = cast(Never, _Provided())
"""The default for a parameter whose ``Annotated`` metadata supplies its value.

``db: Annotated[Connection, Depends(get_db)]`` is a required parameter, so a
caller that type-checks against the function's signature, such as a task
queue that schedules the function with the caller's own arguments, has to pass
it. ``db: Annotated[Connection, Depends(get_db)] = Provided`` makes it
optional for those callers. It is typed ``Never`` so that it fits any
parameter's type. A parameter that defaults to ``Provided`` must have a
``Provider`` in its metadata.
"""
