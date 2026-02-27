"""Tests for annotation-based dependency extraction and resolution."""

from typing import Annotated, Any, cast

import pytest

from uncalled_for import (
    Dependency,
    get_annotation_dependencies,
    resolved_dependencies,
    validate_dependencies,
    without_dependencies,
)


class _Tracker(Dependency["_Tracker"]):
    """Records enter/exit calls for testing."""

    single = True

    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.bound_name: str | None = None
        self.bound_value: Any = None

    def bind_to_parameter(self, name: str, value: Any) -> "_Tracker":
        copy = _Tracker()
        copy.bound_name = name
        copy.bound_value = value
        return copy

    async def __aenter__(self) -> "_Tracker":
        self.entered = True
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited = True


tracker_instance = _Tracker()


# --- get_annotation_dependencies ---


def test_finds_dependency_in_annotated_metadata() -> None:
    dep = _Tracker()

    async def my_func(x: Annotated[int, dep]) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert "x" in result
    assert result["x"] == [dep]


def test_ignores_plain_annotations() -> None:
    async def my_func(x: int, y: str) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert result == {}


def test_ignores_non_dependency_metadata() -> None:
    async def my_func(x: Annotated[int, "not a dep", 42]) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert result == {}


def test_extracts_only_dependency_metadata() -> None:
    dep = _Tracker()

    async def my_func(x: Annotated[int, "label", dep, 99]) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert result["x"] == [dep]


def test_multiple_deps_on_same_parameter() -> None:
    class _DepA(Dependency[str]):
        async def __aenter__(self) -> str: ...

    class _DepB(Dependency[str]):
        async def __aenter__(self) -> str: ...

    async def my_func(x: Annotated[int, _DepA(), _DepB()]) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert len(result["x"]) == 2


def test_deps_on_multiple_parameters() -> None:
    dep_a = _Tracker()
    dep_b = _Tracker()

    async def my_func(
        x: Annotated[int, dep_a],
        y: Annotated[str, dep_b],
    ) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert result["x"] == [dep_a]
    assert result["y"] == [dep_b]


def test_skips_return_annotation() -> None:
    dep = _Tracker()

    async def my_func(x: Annotated[int, dep]) -> Annotated[str, _Tracker()]: ...

    result = get_annotation_dependencies(my_func)
    assert "return" not in result
    assert "x" in result


def test_caches_results() -> None:
    dep = _Tracker()

    async def my_func(x: Annotated[int, dep]) -> None: ...

    first = get_annotation_dependencies(my_func)
    second = get_annotation_dependencies(my_func)
    assert first is second


def test_handles_unresolvable_hints() -> None:
    async def my_func(
        x: "UnresolvableType",  # pyright: ignore[reportUndefinedVariable,reportUnknownParameterType]  # noqa: F821
    ) -> None: ...

    result = get_annotation_dependencies(my_func)  # pyright: ignore[reportUnknownArgumentType]
    assert result == {}


# --- bind_to_parameter ---


def test_default_bind_returns_self() -> None:
    class _Plain(Dependency[str]):
        async def __aenter__(self) -> str: ...

    dep = _Plain()
    bound = dep.bind_to_parameter("x", 42)
    assert bound is dep


def test_subclass_bind_creates_copy() -> None:
    dep = _Tracker()
    bound = dep.bind_to_parameter("customer_id", 99)

    assert bound is not dep
    assert bound.bound_name == "customer_id"
    assert bound.bound_value == 99
    assert dep.bound_name is None


# --- resolved_dependencies with annotations ---


async def test_annotation_deps_entered_and_exited() -> None:
    dep = _Tracker()

    async def my_func(x: Annotated[int, dep]) -> None: ...

    async with resolved_dependencies(my_func, {"x": 42}):
        pass

    # bind_to_parameter returns a copy, so the original is untouched
    assert not dep.entered


async def test_annotation_dep_receives_parameter_value() -> None:
    bound_copies: list[_Tracker] = []

    class _CaptureBind(_Tracker):
        def bind_to_parameter(self, name: str, value: Any) -> "_CaptureBind":
            copy = _CaptureBind()
            copy.bound_name = name
            copy.bound_value = value
            bound_copies.append(copy)
            return copy

    dep = _CaptureBind()

    async def my_func(x: Annotated[int, dep]) -> None: ...

    async with resolved_dependencies(my_func, {"x": 42}):
        assert len(bound_copies) == 1
        assert bound_copies[0].bound_name == "x"
        assert bound_copies[0].bound_value == 42
        assert bound_copies[0].entered


async def test_annotation_dep_value_not_in_arguments() -> None:
    """Annotation deps wrap execution but don't inject values."""
    dep = _Tracker()

    async def my_func(x: Annotated[int, dep]) -> None: ...

    async with resolved_dependencies(my_func, {"x": 42}) as args:
        # Regular parameter x is not in the resolved dependency args —
        # annotation deps wrap execution but don't add to the dict
        assert "x" not in args


async def test_annotation_deps_resolve_after_defaults() -> None:
    """Annotation deps can see values resolved by default deps."""
    bound_values: list[Any] = []

    class _ValueCapture(Dependency["_ValueCapture"]):
        def bind_to_parameter(self, name: str, value: Any) -> "_ValueCapture":
            bound_values.append(value)
            return self

        async def __aenter__(self) -> "_ValueCapture":
            return self

    class _Injector(Dependency[str]):
        async def __aenter__(self) -> str:
            return "injected"

    def Injector() -> str:
        return cast(str, _Injector())

    capture = _ValueCapture()

    async def my_func(
        x: Annotated[str, capture] = Injector(),
    ) -> None: ...

    async with resolved_dependencies(my_func) as args:
        assert args["x"] == "injected"
        assert bound_values == ["injected"]


async def test_annotation_dep_error_propagates() -> None:
    """Annotation dep errors propagate directly, not as FailedDependency."""

    class _Boom(Dependency["_Boom"]):
        async def __aenter__(self) -> "_Boom":
            raise RuntimeError("annotation boom")

    dep = _Boom()

    async def my_func(x: Annotated[int, dep]) -> None: ...

    with pytest.raises(RuntimeError, match="annotation boom"):
        async with resolved_dependencies(my_func, {"x": 1}):
            ...


# --- validate_dependencies with annotations ---


def test_single_annotation_dep_per_param_is_valid() -> None:
    dep = _Tracker()

    async def my_func(x: Annotated[int, dep]) -> None: ...

    validate_dependencies(my_func)


def test_duplicate_single_annotation_dep_on_same_param_raises() -> None:
    async def my_func(
        x: Annotated[int, _Tracker(), _Tracker()],
    ) -> None: ...

    with pytest.raises(
        ValueError,
        match="Only one _Tracker annotation dependency is allowed per parameter",
    ):
        validate_dependencies(my_func)


def test_single_annotation_dep_on_different_params_is_valid() -> None:
    async def my_func(
        x: Annotated[int, _Tracker()],
        y: Annotated[str, _Tracker()],
    ) -> None: ...

    validate_dependencies(my_func)


# --- without_dependencies with annotations ---


async def test_without_dependencies_wraps_annotation_only_functions() -> None:
    entered = False

    class _SideEffect(Dependency["_SideEffect"]):
        async def __aenter__(self) -> "_SideEffect":
            nonlocal entered
            entered = True
            return self

    dep = _SideEffect()

    async def my_func(x: Annotated[int, dep]) -> int:
        return x * 2

    wrapped = without_dependencies(my_func)
    assert wrapped is not my_func  # should be wrapped

    result = await wrapped(x=5)
    assert result == 10
    assert entered
