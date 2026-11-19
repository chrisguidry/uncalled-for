"""A cache keyed by function that lets a dropped function be collected."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from types import MethodType
from typing import Generic, TypeVar

V = TypeVar("V")


class FunctionCache(Generic[V]):
    """Map a callable to a value computed from it, holding the callable weakly.

    A server that builds its tools as closures per request would otherwise
    keep every closure alive, with everything it holds.

    A bound method is a new object on each attribute access, so a bound method
    is keyed by its ``__func__``, apart from the function itself. What is
    computed from a bound method depends only on its function, not on the
    instance it is bound to.

    A callable that cannot be weakly referenced, such as an instance of a class
    with ``__slots__`` and no ``__weakref__``, is held strongly.

    Values are never ``None``, so ``get`` returns ``None`` for a miss. FastMCP
    pops functions from ``introspection._signature_cache`` and
    ``introspection._parameter_cache`` after rewriting their ``__signature__``,
    and docket reports their sizes with ``len()``, so ``pop`` and ``len`` keep
    the behavior of a ``dict``.
    """

    def __init__(self) -> None:
        self._functions: weakref.WeakKeyDictionary[Callable[..., object], V] = (
            weakref.WeakKeyDictionary()
        )
        self._methods: weakref.WeakKeyDictionary[Callable[..., object], V] = (
            weakref.WeakKeyDictionary()
        )
        self._unreferenceable: dict[Callable[..., object], V] = {}

    def get(self, function: Callable[..., object]) -> V | None:
        try:
            if isinstance(function, MethodType):
                return self._methods.get(function.__func__)
            return self._functions.get(function)
        except TypeError:
            return self._unreferenceable.get(function)

    def __setitem__(self, function: Callable[..., object], value: V) -> None:
        try:
            if isinstance(function, MethodType):
                self._methods[function.__func__] = value
            else:
                self._functions[function] = value
        except TypeError:
            self._unreferenceable[function] = value

    def pop(
        self, function: Callable[..., object], default: V | None = None
    ) -> V | None:
        try:
            if isinstance(function, MethodType):
                return self._methods.pop(function.__func__, default)
            return self._functions.pop(function, default)
        except TypeError:
            return self._unreferenceable.pop(function, default)

    def __len__(self) -> int:
        return len(self._functions) + len(self._methods) + len(self._unreferenceable)
