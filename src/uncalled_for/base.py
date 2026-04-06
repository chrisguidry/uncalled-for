"""Base dependency class."""

from __future__ import annotations

import abc
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

T = TypeVar("T", covariant=True)

if TYPE_CHECKING:
    from .classy import _DependencyState


class Dependency(abc.ABC, Generic[T]):
    """Base class for all injectable dependencies.

    Subclasses implement ``__aenter__`` to produce the injected value and
    optionally ``__aexit__`` for cleanup. The resolution engine enters each
    dependency as an async context manager, so resources are cleaned up in
    reverse order when the call completes.

    Set ``single = True`` on a subclass to enforce that only one instance
    of that dependency type may appear in a function's signature.

    Class-level attributes that are ``Dependency`` instances are automatically
    resolved before ``__aenter__`` and cleaned up after ``__aexit__``::

        class MyDep(Dependency[str]):
            pool: Pool = Depends(get_pool)

            async def __aenter__(self) -> str:
                return f"using {self.pool}"
    """

    single: bool = False

    __own_class_dependencies__: ClassVar[dict[str, Dependency[Any]]]
    __class_dependencies__: ClassVar[dict[str, Dependency[Any]]]
    __dependency_state__: _DependencyState

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        from .classy import setup_class_dependencies

        setup_class_dependencies(cls)

    def bind_to_parameter(self, name: str, value: Any) -> Dependency[T]:
        """Return a copy bound to a parameter's name and value.

        Called when the dependency appears as ``Annotated`` metadata.
        Subclasses override to capture context; the default returns *self*.
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
