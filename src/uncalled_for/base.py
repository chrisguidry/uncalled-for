"""Base dependency class."""

from __future__ import annotations

import abc
from types import TracebackType
from typing import Any, Generic, TypeVar

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
