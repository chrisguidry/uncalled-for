"""Tests for concurrent task isolation of class-level dependency attributes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from uncalled_for import Dependency, Depends


async def test_class_dep_values_isolated_across_concurrent_entry() -> None:
    call_count = 0

    def make_value() -> str:
        nonlocal call_count
        call_count += 1
        return f"value-{call_count}"

    class SelfReturning(Dependency["SelfReturning"]):
        value: str = Depends(make_value)

        async def __aenter__(self) -> SelfReturning:
            return self

    shared_instance = SelfReturning()
    entered = asyncio.Event()
    proceed = asyncio.Event()
    values: dict[str, str] = {}

    async def task_a() -> None:
        async with shared_instance as dependency:
            values["a_initial"] = dependency.value
            entered.set()
            await proceed.wait()
            values["a_after_b"] = dependency.value

    async def task_b() -> None:
        await entered.wait()
        async with shared_instance as dependency:
            values["b"] = dependency.value
            proceed.set()

    await asyncio.gather(task_a(), task_b())

    assert values["a_initial"] != values["b"]
    assert values["a_initial"] == values["a_after_b"]


async def test_concurrent_entry_cleanup_is_isolated() -> None:
    cleanups: list[str] = []

    @asynccontextmanager
    async def tracked_resource() -> AsyncGenerator[str]:
        task = asyncio.current_task()
        name = task.get_name() if task else "unknown"
        yield name
        cleanups.append(name)

    class TrackedDependency(Dependency[str]):
        resource: str = Depends(tracked_resource)

        async def __aenter__(self) -> str:
            return self.resource

    shared_instance = TrackedDependency()
    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def task_a() -> None:
        async with shared_instance:
            entered.set()
            await proceed.wait()

    async def task_b() -> None:
        await entered.wait()
        async with shared_instance:
            proceed.set()

    a_task = asyncio.ensure_future(task_a())
    a_task.set_name("a")
    b_task = asyncio.ensure_future(task_b())
    b_task.set_name("b")

    await asyncio.gather(a_task, b_task)

    assert set(cleanups) == {"a", "b"}


async def test_inherited_class_deps_isolated_across_concurrent_entry() -> None:
    call_count = 0

    def make_value() -> str:
        nonlocal call_count
        call_count += 1
        return f"value-{call_count}"

    class Parent(Dependency["Parent"]):
        value: str = Depends(make_value)

        async def __aenter__(self) -> Parent:
            return self

    class Child(Parent):
        pass

    shared_instance = Child()
    entered = asyncio.Event()
    proceed = asyncio.Event()
    values: dict[str, str] = {}

    async def task_a() -> None:
        async with shared_instance as dependency:
            values["a_initial"] = dependency.value
            entered.set()
            await proceed.wait()
            values["a_after_b"] = dependency.value

    async def task_b() -> None:
        await entered.wait()
        async with shared_instance as dependency:
            values["b"] = dependency.value
            proceed.set()

    await asyncio.gather(task_a(), task_b())

    assert values["a_initial"] != values["b"]
    assert values["a_initial"] == values["a_after_b"]


async def test_nested_class_deps_isolated_across_concurrent_entry() -> None:
    call_count = 0

    def make_value() -> str:
        nonlocal call_count
        call_count += 1
        return f"value-{call_count}"

    class Inner(Dependency["Inner"]):
        value: str = Depends(make_value)

        async def __aenter__(self) -> Inner:
            return self

    class Outer(Dependency["Outer"]):
        inner: Inner = Depends(Inner)

        async def __aenter__(self) -> Outer:
            return self

    shared_instance = Outer()
    entered = asyncio.Event()
    proceed = asyncio.Event()
    values: dict[str, str] = {}

    async def task_a() -> None:
        async with shared_instance as dependency:
            values["a_initial"] = dependency.inner.value
            entered.set()
            await proceed.wait()
            values["a_after_b"] = dependency.inner.value

    async def task_b() -> None:
        await entered.wait()
        async with shared_instance as dependency:
            values["b"] = dependency.inner.value
            proceed.set()

    await asyncio.gather(task_a(), task_b())

    assert values["a_initial"] != values["b"]
    assert values["a_initial"] == values["a_after_b"]


async def test_concurrent_error_does_not_corrupt_healthy_task() -> None:
    entered = asyncio.Event()
    may_fail = asyncio.Event()

    def make_value() -> str:
        return "healthy"

    class ErrorDependency(Dependency[str]):
        value: str = Depends(make_value)

        async def __aenter__(self) -> str:
            return self.value

    shared_instance = ErrorDependency()
    healthy_result: str | None = None

    async def healthy_task() -> None:
        nonlocal healthy_result
        async with shared_instance as result:
            healthy_result = result
            entered.set()
            await may_fail.wait()
            healthy_result = result

    async def failing_task() -> None:
        await entered.wait()
        async with shared_instance:
            may_fail.set()
            raise RuntimeError("boom")

    results = await asyncio.gather(
        healthy_task(), failing_task(), return_exceptions=True
    )

    assert healthy_result == "healthy"
    assert isinstance(results[1], RuntimeError)


async def test_many_concurrent_tasks_isolated() -> None:
    call_count = 0

    def make_value() -> str:
        nonlocal call_count
        call_count += 1
        return f"value-{call_count}"

    class SelfReturning(Dependency["SelfReturning"]):
        value: str = Depends(make_value)

        async def __aenter__(self) -> SelfReturning:
            return self

    shared_instance = SelfReturning()
    task_count = 20
    all_entered = asyncio.Event()
    entered_count = 0
    values: dict[int, str] = {}

    async def worker(index: int) -> None:
        nonlocal entered_count
        async with shared_instance as dependency:
            values[index] = dependency.value
            entered_count += 1
            if entered_count == task_count:
                all_entered.set()
            await all_entered.wait()
            assert dependency.value == values[index]

    await asyncio.gather(*(worker(i) for i in range(task_count)))
