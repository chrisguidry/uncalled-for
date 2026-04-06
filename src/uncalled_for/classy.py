"""Class-level dependency scanning, wrapping, and resolution."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any

from .base import Dependency


class _DependencyState:
    """Per-instance state for class dependency lifecycle management."""

    __slots__ = ("stack", "context_reset")

    stack: AsyncExitStack
    context_reset: Callable[[], None] | None

    def __init__(
        self,
        stack: AsyncExitStack,
        context_reset: Callable[[], None] | None,
    ) -> None:
        self.stack = stack
        self.context_reset = context_reset


def _collect_mro_dependencies(
    cls: type[Dependency[Any]],
) -> dict[str, Dependency[Any]]:
    """Walk the MRO and collect all class-level dependency declarations."""
    all_dependencies: dict[str, Dependency[Any]] = {}
    for base in reversed(cls.__mro__):
        if base is Dependency:
            continue
        if issubclass(base, Dependency):
            all_dependencies.update(base.__own_class_dependencies__)
    return all_dependencies


def _unwrap(cls: type[Dependency[Any]], method_name: str, marker: str) -> Any:
    """Find the user-defined method by unwrapping any framework wrappers."""
    for klass in cls.__mro__:
        method = klass.__dict__.get(method_name)
        if method is not None:
            while hasattr(method, marker):
                method = getattr(method, marker)
            return method
    raise TypeError(f"{cls.__name__} has no {method_name} in its MRO")


def _ensure_depends_context(
    stack: AsyncExitStack,
) -> Callable[[], None] | None:
    """Set up ``_Depends`` ContextVars if not already set.

    Returns a reset callback, or ``None`` if the context was already active.
    """
    from .functional import _Depends

    try:
        _Depends.cache.get()
        return None
    except LookupError:
        cache_token = _Depends.cache.set({})
        stack_token = _Depends.stack.set(stack)

        def reset() -> None:
            _Depends.stack.reset(stack_token)
            _Depends.cache.reset(cache_token)

        return reset


def _make_wrapped_aenter(
    dependencies: dict[str, Dependency[Any]],
    original_aenter: Any,
) -> Any:
    """Build a ``__aenter__`` wrapper that resolves class dependencies."""

    async def wrapped_aenter(self: Dependency[Any]) -> Any:
        stack = AsyncExitStack()
        await stack.__aenter__()
        context_reset = _ensure_depends_context(stack)

        try:
            for name, dependency in dependencies.items():
                setattr(self, name, await stack.enter_async_context(dependency))
            result = await original_aenter(self)
        except BaseException:
            if context_reset:
                context_reset()
            await stack.__aexit__(None, None, None)
            raise

        self.__dependency_state__ = _DependencyState(stack, context_reset)
        return result

    wrapped_aenter.__original_aenter__ = original_aenter  # type: ignore[attr-defined]
    return wrapped_aenter


def _make_wrapped_aexit(original_aexit: Any) -> Any:
    """Build an ``__aexit__`` wrapper that cleans up class dependencies."""

    async def wrapped_aexit(
        self: Dependency[Any],
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await original_aexit(self, exc_type, exc_value, traceback)
        finally:
            state = self.__dependency_state__
            try:
                await state.stack.__aexit__(None, None, None)
            finally:
                if state.context_reset:
                    state.context_reset()

    wrapped_aexit.__original_aexit__ = original_aexit  # type: ignore[attr-defined]
    return wrapped_aexit


def setup_class_dependencies(cls: type[Dependency[Any]]) -> None:
    """Scan for class-level dependencies and wire up resolution."""
    own_dependencies: dict[str, Dependency[Any]] = {
        name: value
        for name, value in cls.__dict__.items()
        if isinstance(value, Dependency)
    }

    cls.__own_class_dependencies__ = own_dependencies

    for name in own_dependencies:
        delattr(cls, name)

    all_dependencies = _collect_mro_dependencies(cls)

    if not all_dependencies:
        return

    cls.__class_dependencies__ = dict(all_dependencies)

    if not own_dependencies and "__aenter__" not in cls.__dict__:
        return

    original_aenter = _unwrap(cls, "__aenter__", "__original_aenter__")
    original_aexit = _unwrap(cls, "__aexit__", "__original_aexit__")

    cls.__aenter__ = _make_wrapped_aenter(  # type: ignore[attr-defined]
        all_dependencies, original_aenter
    )
    cls.__aexit__ = _make_wrapped_aexit(original_aexit)  # type: ignore[attr-defined]
