"""Factory-based dependencies: Depends and its internals."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Hashable
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
)
from contextvars import ContextVar
from typing import Any, ClassVar, TypeVar, cast, overload

from .base import Dependency
from .frames import _CallArgument, _stack  # pyright: ignore[reportPrivateUsage]
from .introspection import get_dependency_parameters

R = TypeVar("R")

DependencyFactory = Callable[
    ..., R | Awaitable[R] | AbstractContextManager[R] | AbstractAsyncContextManager[R]
]

CacheKey = (
    DependencyFactory[Any]
    | tuple[DependencyFactory[Any], tuple[tuple[str, Hashable], ...]]
)


class _FunctionalDependency(Dependency[R]):
    """Base for dependencies that wrap a factory function."""

    factory: DependencyFactory[R]

    def __init__(self, factory: DependencyFactory[R]) -> None:
        self.factory = factory

    async def _resolve_factory_value(
        self,
        stack: AsyncExitStack,
        raw_value: (
            R
            | Awaitable[R]
            | AbstractContextManager[R]
            | AbstractAsyncContextManager[R]
        ),
    ) -> R:
        if isinstance(raw_value, AbstractAsyncContextManager):
            return await stack.enter_async_context(raw_value)  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(raw_value, AbstractContextManager):
            return stack.enter_context(raw_value)  # pyright: ignore[reportUnknownArgumentType]
        elif inspect.iscoroutine(raw_value) or isinstance(raw_value, Awaitable):
            return await cast(Awaitable[R], raw_value)
        else:
            return cast(R, raw_value)


class _Depends(_FunctionalDependency[R]):
    """Call-scoped dependency, resolved fresh for each call."""

    cache: ClassVar[ContextVar[dict[CacheKey, Any]]] = ContextVar("uncalled_for_cache")
    stack: ClassVar[ContextVar[AsyncExitStack]] = _stack

    bindings: dict[str, Any]

    def __init__(self, factory: DependencyFactory[R], **bindings: Any) -> None:
        super().__init__(factory)
        self.bindings = bindings

    def _cache_key(self) -> CacheKey:
        if not self.bindings:
            return self.factory

        return (
            self.factory,
            tuple(
                (name, _binding_key(self.bindings[name], name))
                for name in sorted(self.bindings)
            ),
        )

    async def _resolve_parameters(
        self,
        function: Callable[..., Any],
    ) -> dict[str, Any]:
        stack = self.stack.get()
        arguments: dict[str, Any] = {}
        parameters = get_dependency_parameters(function)

        for parameter, dependency in parameters.items():
            if parameter in self.bindings:
                continue

            arguments[parameter] = await stack.enter_async_context(
                dependency.for_parameter(parameter)
            )

        return arguments

    async def __aenter__(self) -> R:
        cache = self.cache.get()
        key = self._cache_key()

        if key in cache:
            return cast(R, cache[key])

        stack = self.stack.get()
        arguments = await self._resolve_parameters(self.factory)

        for name, value in self.bindings.items():
            if isinstance(value, Dependency):
                dependency = cast(Dependency[Any], value)
                arguments[name] = await stack.enter_async_context(
                    dependency.for_parameter(name)
                )
            else:
                arguments[name] = value

        raw_value = self.factory(**arguments)
        resolved_value = await self._resolve_factory_value(stack, raw_value)

        cache[key] = resolved_value
        return resolved_value


def _binding_key(value: Any, name: str) -> Hashable:
    """Produce a hashable cache key for the binding *value* on parameter *name*.

    Two bindings share a cached factory result only when their keys are equal.
    The ``"value"`` and ``"id"`` tags keep a tuple a caller passed from
    matching a key this function built.
    """
    if isinstance(value, _CallArgument):
        # A bare CallArgument resolves the parameter it is bound to, so its
        # key uses that name and matches the equivalent explicit form. The
        # runtime type is part of the key so that a subclass with different
        # semantics never shares an entry with the class it extends.
        return (type(value), value.parameter or name, value.optional)

    if isinstance(value, _Depends):
        depends = cast("_Depends[Any]", value)
        return depends._cache_key()  # pyright: ignore[reportPrivateUsage]

    if isinstance(value, Dependency):
        # Two custom dependencies that compare equal may still do different
        # work on entry, so identity is the only safe key. The instance stays
        # alive in the bindings, so its id cannot be reused.
        dependency = cast(Dependency[Any], value)
        return ("id", id(dependency))

    try:
        hash(value)
    except TypeError:
        return ("id", id(value))

    # The runtime type is part of the key because Python compares 1, 1.0, and
    # True as equal, and a factory can produce different results for each.
    return ("value", cast(type[Any], type(value)), value)


@overload
def Depends(
    factory: Callable[..., AbstractAsyncContextManager[R]], **bindings: Any
) -> R: ...
@overload
def Depends(
    factory: Callable[..., AbstractContextManager[R]], **bindings: Any
) -> R: ...
@overload
def Depends(factory: Callable[..., Awaitable[R]], **bindings: Any) -> R: ...
@overload
def Depends(factory: Callable[..., R], **bindings: Any) -> R: ...
def Depends(factory: DependencyFactory[R], **bindings: Any) -> R:
    """Declare a dependency on a factory function.

    The factory is called once per resolution scope. It may be:

    - A sync function returning a value
    - An async function returning a value
    - A sync generator (context manager) yielding a value
    - An async generator (async context manager) yielding a value

    Context managers get proper enter/exit lifecycle management.

    Keyword *bindings* supply arguments to the factory from the place that
    declares the dependency. A ``Dependency`` value, such as ``CallArgument()``
    or another ``Depends``, resolves first and the factory receives its value.
    Any other value passes through as it is. A binding replaces the default of
    the factory's own parameter, which is then never resolved. Two dependencies
    on the same factory share a cached result only when their bindings match.

    Example::

        def get_account(user_id: str, db: Database = Depends(get_db)) -> Account:
            return db.accounts[user_id]

        async def show(
            owner: str,
            account: Account = Depends(get_account, user_id=CallArgument("owner")),
        ):
            ...
    """
    return cast(R, _Depends(factory, **bindings))
