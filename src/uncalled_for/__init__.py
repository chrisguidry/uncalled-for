"""Async dependency injection for Python functions.

Declare dependencies as parameter defaults. They resolve automatically when
the function is called through the dependency resolution context manager.
"""

from __future__ import annotations

import abc
import asyncio
import inspect
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable
from functools import lru_cache
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from contextvars import ContextVar
from types import TracebackType
from typing import Any, ClassVar, Generic, TypeVar, cast, overload

T = TypeVar("T", covariant=True)
R = TypeVar("R")

DependencyFactory = Callable[
    ..., R | Awaitable[R] | AbstractContextManager[R] | AbstractAsyncContextManager[R]
]


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

    @abc.abstractmethod
    async def __aenter__(self) -> T: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass


_parameter_cache: dict[Callable[..., Any], dict[str, Dependency[Any]]] = {}


def get_dependency_parameters(
    function: Callable[..., Any],
) -> dict[str, Dependency[Any]]:
    """Find parameters whose defaults are Dependency instances."""
    if function in _parameter_cache:
        return _parameter_cache[function]

    dependencies: dict[str, Dependency[Any]] = {}
    signature = get_signature(function)

    for name, param in signature.parameters.items():
        if isinstance(param.default, Dependency):
            dependencies[name] = param.default  # pyright: ignore[reportUnknownMemberType]

    _parameter_cache[function] = dependencies
    return dependencies


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

    cache: ClassVar[ContextVar[dict[DependencyFactory[Any], Any]]] = ContextVar(
        "uncalled_for_cache"
    )
    stack: ClassVar[ContextVar[AsyncExitStack]] = ContextVar("uncalled_for_stack")

    async def _resolve_parameters(
        self,
        function: Callable[..., Any],
    ) -> dict[str, Any]:
        stack = self.stack.get()
        arguments: dict[str, Any] = {}
        parameters = get_dependency_parameters(function)

        for parameter, dependency in parameters.items():
            arguments[parameter] = await stack.enter_async_context(dependency)

        return arguments

    async def __aenter__(self) -> R:
        cache = self.cache.get()

        if self.factory in cache:
            return cache[self.factory]

        stack = self.stack.get()
        arguments = await self._resolve_parameters(self.factory)
        raw_value = self.factory(**arguments)
        resolved_value = await self._resolve_factory_value(stack, raw_value)

        cache[self.factory] = resolved_value
        return resolved_value


@overload
def Depends(factory: Callable[..., AbstractAsyncContextManager[R]]) -> R: ...
@overload
def Depends(factory: Callable[..., AbstractContextManager[R]]) -> R: ...
@overload
def Depends(factory: Callable[..., Awaitable[R]]) -> R: ...
@overload
def Depends(factory: Callable[..., R]) -> R: ...
def Depends(factory: DependencyFactory[R]) -> R:
    """Declare a dependency on a factory function.

    The factory is called once per resolution scope. It may be:

    - A sync function returning a value
    - An async function returning a value
    - A sync generator (context manager) yielding a value
    - An async generator (async context manager) yielding a value

    Context managers get proper enter/exit lifecycle management.
    """
    return cast(R, _Depends(factory))


class _Shared(_FunctionalDependency[R]):
    """App-scoped dependency resolved once and reused across all calls.

    Unlike _Depends (which resolves per-call), _Shared dependencies initialize
    once within a SharedContext and the same instance is provided to all
    subsequent resolutions.
    """

    async def __aenter__(self) -> R:
        resolved = SharedContext.resolved.get()

        if self.factory in resolved:
            return resolved[self.factory]

        arguments = await self._resolve_parameters()

        async with SharedContext.lock.get():
            if self.factory in resolved:  # pragma: no cover
                return resolved[self.factory]

            stack = SharedContext.stack.get()
            raw_value = self.factory(**arguments)
            resolved_value = await self._resolve_factory_value(stack, raw_value)

            resolved[self.factory] = resolved_value
            return resolved_value

    async def _resolve_parameters(self) -> dict[str, Any]:
        stack = SharedContext.stack.get()
        arguments: dict[str, Any] = {}
        parameters = get_dependency_parameters(self.factory)

        for parameter, dependency in parameters.items():
            arguments[parameter] = await stack.enter_async_context(dependency)

        return arguments


class SharedContext:
    """Manages app-scoped Shared dependency lifecycle.

    Use as an async context manager to establish a scope for Shared
    dependencies. All Shared factories resolved within this scope will
    be cached and reused. Context managers are cleaned up when the
    SharedContext exits.

    Example::

        async with SharedContext():
            async with resolved_dependencies(my_func) as deps:
                # Shared deps are resolved once and cached here
                ...
            async with resolved_dependencies(my_func) as deps:
                # Same Shared instances reused
                ...
        # Shared context managers are cleaned up here
    """

    resolved: ClassVar[ContextVar[dict[DependencyFactory[Any], Any]]] = ContextVar(
        "shared_resolved"
    )
    lock: ClassVar[ContextVar[asyncio.Lock]] = ContextVar("shared_lock")
    stack: ClassVar[ContextVar[AsyncExitStack]] = ContextVar("shared_stack")

    async def __aenter__(self) -> SharedContext:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        self._resolved_token = SharedContext.resolved.set({})
        self._lock_token = SharedContext.lock.set(asyncio.Lock())
        self._stack_token = SharedContext.stack.set(self._stack)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._stack.__aexit__(exc_type, exc_value, traceback)

        SharedContext.stack.reset(self._stack_token)
        SharedContext.lock.reset(self._lock_token)
        SharedContext.resolved.reset(self._resolved_token)


@overload
def Shared(factory: Callable[..., AbstractAsyncContextManager[R]]) -> R: ...
@overload
def Shared(factory: Callable[..., AbstractContextManager[R]]) -> R: ...
@overload
def Shared(factory: Callable[..., Awaitable[R]]) -> R: ...
@overload
def Shared(factory: Callable[..., R]) -> R: ...
def Shared(factory: DependencyFactory[R]) -> R:
    """Declare an app-scoped dependency shared across all calls.

    The factory initializes once within a ``SharedContext`` and the value is
    reused for all subsequent resolutions. Factories may be:

    - A sync function returning a value
    - An async function returning a value
    - A sync generator (context manager) yielding a value
    - An async generator (async context manager) yielding a value

    Context managers are cleaned up when the SharedContext exits.
    Identity is the factory function — multiple ``Shared(same_factory)``
    declarations anywhere resolve to the same cached value.
    """
    return cast(R, _Shared(factory))


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

    # Phase 1: check for duplicate concrete types.  This catches e.g. two
    # Retry(...) defaults and reports "Only one Retry dependency is allowed".
    counts: Counter[type[Dependency[Any]]] = Counter(
        type(d)
        for d in dependencies  # pyright: ignore[reportUnknownArgumentType]
    )
    for dep_type, count in counts.items():
        if getattr(dep_type, "single", False) and count > 1:  # pyright: ignore[reportUnknownArgumentType]
            raise ValueError(
                f"Only one {dep_type.__name__} dependency is allowed"  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
            )

    # Phase 2: check for conflicts between *different* subclasses that share
    # a single base (e.g. Timeout + CustomRuntime both under Runtime).
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
        instances = [d for d in dependencies if isinstance(d, base_class)]
        if len(instances) > 1:
            types = ", ".join(type(d).__name__ for d in instances)
            raise ValueError(
                f"Only one {base_class.__name__} dependency is allowed, "
                f"but found: {types}"
            )


class FailedDependency:
    """Placeholder for a dependency that raised during resolution."""

    def __init__(self, parameter: str, error: Exception) -> None:
        self.parameter = parameter
        self.error = error


@asynccontextmanager
async def resolved_dependencies(
    function: Callable[..., Any],
    kwargs: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any]]:
    """Resolve all dependencies declared on a function's signature.

    Yields a dict mapping parameter names to resolved values. Dependencies
    are entered as async context managers and cleaned up when the context
    exits.

    Parameters already present in *kwargs* are passed through without
    resolution, allowing callers to override specific dependencies.
    """
    provided = kwargs or {}
    cache_token = _Depends.cache.set({})

    try:
        async with AsyncExitStack() as stack:
            stack_token = _Depends.stack.set(stack)
            try:
                arguments: dict[str, Any] = {}
                parameters = get_dependency_parameters(function)

                for parameter, dependency in parameters.items():
                    if parameter in provided:
                        arguments[parameter] = provided[parameter]
                        continue

                    try:
                        arguments[parameter] = await stack.enter_async_context(
                            dependency
                        )
                    except Exception as error:
                        arguments[parameter] = FailedDependency(parameter, error)

                yield arguments
            finally:
                _Depends.stack.reset(stack_token)
    finally:
        _Depends.cache.reset(cache_token)


@lru_cache(maxsize=5_000)
def without_dependencies(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Produce a wrapper whose signature hides dependency parameters.

    If *fn* has no ``Dependency`` defaults, it is returned unchanged.
    Otherwise an async wrapper is returned that resolves dependencies
    automatically and forwards user-supplied keyword arguments.
    """
    dep_names = set(get_dependency_parameters(fn))
    if not dep_names:
        return fn

    original_sig = get_signature(fn)
    filtered_params = [
        p for name, p in original_sig.parameters.items() if name not in dep_names
    ]
    new_sig = original_sig.replace(
        parameters=filtered_params, return_annotation=inspect.Parameter.empty
    )

    is_async = inspect.iscoroutinefunction(fn)

    async def wrapper(**kwargs: Any) -> Any:
        async with resolved_dependencies(fn, kwargs) as resolved:
            all_kwargs = {**resolved, **kwargs}
            if is_async:
                return await fn(**all_kwargs)
            return fn(**all_kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    wrapper.__annotations__ = {
        k: v
        for k, v in fn.__annotations__.items()
        if k not in dep_names and k != "return"
    }

    return wrapper


__all__ = [
    "Dependency",
    "DependencyFactory",
    "Depends",
    "FailedDependency",
    "Shared",
    "SharedContext",
    "get_dependency_parameters",
    "get_signature",
    "resolved_dependencies",
    "validate_dependencies",
    "without_dependencies",
]
