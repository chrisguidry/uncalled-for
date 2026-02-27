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


class Tracker(Dependency["Tracker"]):
    """Records enter/exit calls for testing."""

    single = True

    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.bound_name: str | None = None
        self.bound_value: Any = None

    def bind_to_parameter(self, name: str, value: Any) -> "Tracker":
        copy = Tracker()
        copy.bound_name = name
        copy.bound_value = value
        return copy

    async def __aenter__(self) -> "Tracker":
        self.entered = True
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited = True


tracker_instance = Tracker()


def test_finds_dependency_in_annotated_metadata() -> None:
    dependency = Tracker()

    async def my_func(x: Annotated[int, dependency]) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert "x" in result
    assert result["x"] == [dependency]


def test_ignores_plain_annotations() -> None:
    async def my_func(x: int, y: str) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert result == {}


def test_ignores_non_dependency_metadata() -> None:
    async def my_func(x: Annotated[int, "not a dep", 42]) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert result == {}


def test_extracts_only_dependency_metadata() -> None:
    dependency = Tracker()

    async def my_func(x: Annotated[int, "label", dependency, 99]) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert result["x"] == [dependency]


def test_multiple_dependencies_on_same_parameter() -> None:
    class DependencyA(Dependency[str]):
        async def __aenter__(self) -> str: ...

    class DependencyB(Dependency[str]):
        async def __aenter__(self) -> str: ...

    async def my_func(x: Annotated[int, DependencyA(), DependencyB()]) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert len(result["x"]) == 2


def test_dependencies_on_multiple_parameters() -> None:
    first_dependency = Tracker()
    second_dependency = Tracker()

    async def my_func(
        x: Annotated[int, first_dependency],
        y: Annotated[str, second_dependency],
    ) -> None: ...

    result = get_annotation_dependencies(my_func)
    assert result["x"] == [first_dependency]
    assert result["y"] == [second_dependency]


def test_skips_return_annotation() -> None:
    dependency = Tracker()

    async def my_func(
        x: Annotated[int, dependency],
    ) -> Annotated[str, Tracker()]: ...

    result = get_annotation_dependencies(my_func)
    assert "return" not in result
    assert "x" in result


def test_caches_results() -> None:
    dependency = Tracker()

    async def my_func(x: Annotated[int, dependency]) -> None: ...

    first = get_annotation_dependencies(my_func)
    second = get_annotation_dependencies(my_func)
    assert first is second


def test_handles_unresolvable_hints() -> None:
    async def my_func(
        x: "UnresolvableType",  # pyright: ignore[reportUndefinedVariable,reportUnknownParameterType]  # noqa: F821
    ) -> None: ...

    result = get_annotation_dependencies(my_func)  # pyright: ignore[reportUnknownArgumentType]
    assert result == {}


def test_default_bind_returns_self() -> None:
    class PlainDependency(Dependency[str]):
        async def __aenter__(self) -> str: ...

    dependency = PlainDependency()
    bound = dependency.bind_to_parameter("x", 42)
    assert bound is dependency


def test_subclass_bind_creates_copy() -> None:
    dependency = Tracker()
    bound = dependency.bind_to_parameter("customer_id", 99)

    assert bound is not dependency
    assert bound.bound_name == "customer_id"
    assert bound.bound_value == 99
    assert dependency.bound_name is None


async def test_annotation_dependency_entered_and_exited() -> None:
    dependency = Tracker()

    async def my_func(x: Annotated[int, dependency]) -> None: ...

    async with resolved_dependencies(my_func, {"x": 42}):
        pass

    # bind_to_parameter returns a copy, so the original is untouched
    assert not dependency.entered


async def test_annotation_dependency_receives_parameter_value() -> None:
    bound_copies: list[Tracker] = []

    class CapturingDependency(Tracker):
        def bind_to_parameter(self, name: str, value: Any) -> "CapturingDependency":
            copy = CapturingDependency()
            copy.bound_name = name
            copy.bound_value = value
            bound_copies.append(copy)
            return copy

    dependency = CapturingDependency()

    async def my_func(x: Annotated[int, dependency]) -> None: ...

    async with resolved_dependencies(my_func, {"x": 42}):
        assert len(bound_copies) == 1
        assert bound_copies[0].bound_name == "x"
        assert bound_copies[0].bound_value == 42
        assert bound_copies[0].entered


async def test_annotation_dependency_value_not_in_arguments() -> None:
    """Annotation dependencies wrap execution but don't inject values."""
    dependency = Tracker()

    async def my_func(x: Annotated[int, dependency]) -> None: ...

    async with resolved_dependencies(my_func, {"x": 42}) as arguments:
        assert "x" not in arguments


async def test_annotation_dependencies_resolve_after_defaults() -> None:
    """Annotation dependencies can see values resolved by default dependencies."""
    bound_values: list[Any] = []

    class ValueCapture(Dependency["ValueCapture"]):
        def bind_to_parameter(self, name: str, value: Any) -> "ValueCapture":
            bound_values.append(value)
            return self

        async def __aenter__(self) -> "ValueCapture":
            return self

    class Injector(Dependency[str]):
        async def __aenter__(self) -> str:
            return "injected"

    def make_injector() -> str:
        return cast(str, Injector())

    capture = ValueCapture()

    async def my_func(
        x: Annotated[str, capture] = make_injector(),
    ) -> None: ...

    async with resolved_dependencies(my_func) as arguments:
        assert arguments["x"] == "injected"
        assert bound_values == ["injected"]


async def test_annotation_dependency_error_propagates() -> None:
    """Annotation dependency errors propagate directly, not as FailedDependency."""

    class ExplodingDependency(Dependency["ExplodingDependency"]):
        async def __aenter__(self) -> "ExplodingDependency":
            raise RuntimeError("annotation boom")

    dependency = ExplodingDependency()

    async def my_func(x: Annotated[int, dependency]) -> None: ...

    with pytest.raises(RuntimeError, match="annotation boom"):
        async with resolved_dependencies(my_func, {"x": 1}):
            ...


def test_single_annotation_dependency_per_parameter_is_valid() -> None:
    dependency = Tracker()

    async def my_func(x: Annotated[int, dependency]) -> None: ...

    assert validate_dependencies(my_func) is None


def test_duplicate_single_annotation_dependency_on_same_parameter_raises() -> None:
    async def my_func(
        x: Annotated[int, Tracker(), Tracker()],
    ) -> None: ...

    with pytest.raises(
        ValueError,
        match="Only one Tracker annotation dependency is allowed per parameter",
    ):
        validate_dependencies(my_func)


def test_single_annotation_dependency_on_different_parameters_is_valid() -> None:
    async def my_func(
        x: Annotated[int, Tracker()],
        y: Annotated[str, Tracker()],
    ) -> None: ...

    assert validate_dependencies(my_func) is None


async def test_without_dependencies_wraps_annotation_only_functions() -> None:
    entered = False

    class SideEffectDependency(Dependency["SideEffectDependency"]):
        async def __aenter__(self) -> "SideEffectDependency":
            nonlocal entered
            entered = True
            return self

    dependency = SideEffectDependency()

    async def my_func(x: Annotated[int, dependency]) -> int:
        return x * 2

    wrapped = without_dependencies(my_func)
    assert wrapped is not my_func

    result = await wrapped(x=5)
    assert result == 10
    assert entered
