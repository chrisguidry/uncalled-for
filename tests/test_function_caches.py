"""Tests for the caches keyed by function."""

from __future__ import annotations

import gc
import inspect
import weakref
from typing import Annotated, Any

from uncalled_for import (
    Dependency,
    Depends,
    get_annotation_dependencies,
    get_call_argument_parameters,
    get_dependency_parameters,
    get_signature,
    introspection,
    resolved_dependencies,
)


class Payload:
    pass


class _Traced(Dependency[None]):
    async def __aenter__(self) -> None: ...


async def _resolve_dropped_handler() -> tuple[weakref.ref[Any], weakref.ref[Payload]]:
    payload = Payload()

    def get_payload() -> Payload:
        return payload

    async def handler(
        value: Annotated[Payload, _Traced()] = Depends(get_payload),
    ) -> Payload: ...

    async with resolved_dependencies(handler) as arguments:
        assert arguments["value"] is payload
    assert get_annotation_dependencies(handler)
    assert get_call_argument_parameters(handler) == []

    return weakref.ref(handler), weakref.ref(payload)


async def test_a_dropped_function_and_its_closure_are_collected() -> None:
    handler, payload = await _resolve_dropped_handler()
    gc.collect()

    assert handler() is None
    assert payload() is None


class Greeter:
    def greet(self, name: str = Depends(lambda: "world")) -> str: ...


def test_a_bound_method_hits_the_cache_on_each_attribute_access() -> None:
    greeter = Greeter()

    assert get_signature(greeter.greet) is get_signature(greeter.greet)
    assert get_dependency_parameters(greeter.greet) is get_dependency_parameters(
        greeter.greet
    )
    assert get_signature(greeter.greet) is get_signature(Greeter().greet)


def test_a_bound_method_and_its_function_are_cached_apart() -> None:
    greeter = Greeter()

    assert list(get_signature(greeter.greet).parameters) == ["name"]
    assert list(get_signature(Greeter.greet).parameters) == ["self", "name"]


class Unreferenceable:
    __slots__ = ()

    def __call__(self, x: int) -> None: ...


def test_a_callable_without_weak_references_is_cached() -> None:
    function = Unreferenceable()

    assert get_signature(function) is get_signature(function)
    assert get_dependency_parameters(function) is get_dependency_parameters(function)
    assert introspection._signature_cache.pop(function, None) is not None  # pyright: ignore[reportPrivateUsage]
    assert introspection._signature_cache.pop(function, None) is None  # pyright: ignore[reportPrivateUsage]


class Unhashable:
    __hash__ = None  # type: ignore[assignment]

    def method(self, x: int) -> None: ...


def test_a_method_of_an_unhashable_instance_is_cached() -> None:
    instance = Unhashable()

    assert get_signature(instance.method) is get_signature(instance.method)


def test_popping_a_function_clears_its_cached_signature() -> None:
    """FastMCP rewrites ``__signature__`` and then pops the function from
    ``_signature_cache`` and ``_parameter_cache``, and docket reports their
    sizes with ``len()``."""

    def my_func(x: int) -> None: ...

    get_signature(my_func)
    get_dependency_parameters(my_func)
    assert len(introspection._signature_cache) > 0  # pyright: ignore[reportPrivateUsage]
    assert len(introspection._parameter_cache) > 0  # pyright: ignore[reportPrivateUsage]

    my_func.__signature__ = inspect.Signature()  # pyright: ignore[reportFunctionMemberAccess]
    assert introspection._signature_cache.pop(my_func, None) is not None  # pyright: ignore[reportPrivateUsage]
    assert introspection._parameter_cache.pop(my_func, None) is not None  # pyright: ignore[reportPrivateUsage]

    assert list(get_signature(my_func).parameters) == []


def test_popping_a_bound_method_clears_its_cached_signature() -> None:
    class Tools:
        def search(self, query: str) -> None: ...

    method = Tools().search
    cached = get_signature(method)

    Tools.search.__signature__ = inspect.Signature()  # pyright: ignore[reportFunctionMemberAccess]
    assert introspection._signature_cache.pop(method, None) is cached  # pyright: ignore[reportPrivateUsage]

    assert get_signature(method) is not cached
