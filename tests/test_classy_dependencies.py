"""Tests for class-level dependency declarations on Dependency subclasses."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

import pytest

from uncalled_for.classy import _unwrap  # pyright: ignore[reportPrivateUsage]

from uncalled_for import (
    Dependency,
    Depends,
    FailedDependency,
    Shared,
    SharedContext,
    get_class_dependencies,
    resolved_dependencies,
    without_dependencies,
)


class Greeter(Dependency[str]):
    async def __aenter__(self) -> str: ...


def test_class_without_class_deps_unchanged() -> None:
    assert not hasattr(Greeter, "__class_dependencies__")
    assert get_class_dependencies(Greeter) == {}


async def test_basic_depends_class_attr() -> None:
    def get_value() -> str:
        return "resolved-value"

    class MyDep(Dependency[str]):
        value: str = Depends(get_value)

        async def __aenter__(self) -> str:
            return f"got {self.value}"

    async with MyDep() as result:
        assert result == "got resolved-value"


async def test_bare_dependency_subclass_as_class_attr() -> None:
    class Inner(Dependency[int]):
        async def __aenter__(self) -> int:
            return 42

    class Outer(Dependency[str]):
        inner: int = Inner()  # type: ignore[assignment]

        async def __aenter__(self) -> str:
            return f"inner={self.inner}"

    async with Outer() as result:
        assert result == "inner=42"


async def test_self_typed_dependency_as_class_attr() -> None:
    class Config(Dependency["Config"]):
        def __init__(self, url: str = "http://default") -> None:
            self.url = url

        async def __aenter__(self) -> Config:
            return self

    class Service(Dependency[str]):
        config: Config = Config()

        async def __aenter__(self) -> str:
            return f"service({self.config.url})"

    async with Service() as result:
        assert result == "service(http://default)"


async def test_multiple_class_level_deps() -> None:
    def get_a() -> str:
        return "a"

    def get_b() -> int:
        return 2

    class Multi(Dependency[str]):
        a: str = Depends(get_a)
        b: int = Depends(get_b)

        async def __aenter__(self) -> str:
            return f"{self.a}-{self.b}"

    async with Multi() as result:
        assert result == "a-2"


async def test_lifecycle_ordering() -> None:
    order: list[str] = []

    @asynccontextmanager
    async def resource_a() -> AsyncGenerator[str]:
        order.append("a-enter")
        yield "a"
        order.append("a-exit")

    @asynccontextmanager
    async def resource_b() -> AsyncGenerator[str]:
        order.append("b-enter")
        yield "b"
        order.append("b-exit")

    class LifecycleDep(Dependency[str]):
        a: str = Depends(resource_a)
        b: str = Depends(resource_b)

        async def __aenter__(self) -> str:
            order.append("enter")
            return f"{self.a},{self.b}"

        async def __aexit__(self, *args: object) -> None:
            order.append("exit")

    async with LifecycleDep() as result:
        assert result == "a,b"

    assert order == ["a-enter", "b-enter", "enter", "exit", "b-exit", "a-exit"]


async def test_integration_with_resolved_dependencies() -> None:
    def get_connection() -> str:
        return "db-conn"

    class DbDep(Dependency[str]):
        conn: str = Depends(get_connection)

        async def __aenter__(self) -> str:
            return f"using {self.conn}"

    def make_db_dep() -> str:
        return cast(str, DbDep())

    async def my_func(db: str = make_db_dep()) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert deps["db"] == "using db-conn"


async def test_integration_with_without_dependencies() -> None:
    def get_connection() -> str:
        return "db-conn"

    class DbDep(Dependency[str]):
        conn: str = Depends(get_connection)

        async def __aenter__(self) -> str:
            return f"using {self.conn}"

    def make_db_dep() -> str:
        return cast(str, DbDep())

    async def my_func(db: str = make_db_dep()) -> str:
        return db

    wrapped = without_dependencies(my_func)
    result = await wrapped()
    assert result == "using db-conn"


async def test_shared_class_attr() -> None:
    call_count = 0

    def make_pool() -> str:
        nonlocal call_count
        call_count += 1
        return "shared-pool"

    class PoolDep(Dependency[str]):
        pool: str = Shared(make_pool)

        async def __aenter__(self) -> str:
            return self.pool

    def make_pool_dep() -> str:
        return cast(str, PoolDep())

    async def func_a(v: str = make_pool_dep()) -> None: ...
    async def func_b(v: str = make_pool_dep()) -> None: ...

    async with SharedContext():
        async with resolved_dependencies(func_a) as deps_a:
            assert deps_a["v"] == "shared-pool"

        async with resolved_dependencies(func_b) as deps_b:
            assert deps_b["v"] == "shared-pool"

        assert call_count == 1


async def test_error_in_class_dep_propagates() -> None:
    class Broken(Dependency[str]):
        async def __aenter__(self) -> str:
            raise RuntimeError("broken dep")

    class Consumer(Dependency[str]):
        broken: str = Depends(Broken)

        async def __aenter__(self) -> str: ...

    with pytest.raises(RuntimeError, match="broken dep"):
        async with Consumer():
            ...


async def test_error_in_class_dep_cleans_up_others() -> None:
    cleaned_up = False

    @asynccontextmanager
    async def good_resource() -> AsyncGenerator[str]:
        yield "good"
        nonlocal cleaned_up
        cleaned_up = True

    class Broken(Dependency[str]):
        async def __aenter__(self) -> str:
            raise RuntimeError("broken")

    class Consumer(Dependency[str]):
        good: str = Depends(good_resource)
        broken: str = Depends(Broken)

        async def __aenter__(self) -> str: ...

    with pytest.raises(RuntimeError, match="broken"):
        async with Consumer():
            ...

    assert cleaned_up


async def test_error_in_aenter_cleans_up_deps() -> None:
    cleaned_up = False

    @asynccontextmanager
    async def managed_resource() -> AsyncGenerator[str]:
        yield "resource"
        nonlocal cleaned_up
        cleaned_up = True

    class FailingDep(Dependency[str]):
        resource: str = Depends(managed_resource)

        async def __aenter__(self) -> str:
            raise RuntimeError("aenter failed")

    with pytest.raises(RuntimeError, match="aenter failed"):
        async with FailingDep():
            ...

    assert cleaned_up


async def test_error_inside_resolved_dependencies_context() -> None:
    class Broken(Dependency[str]):
        async def __aenter__(self) -> str:
            raise RuntimeError("boom")

    class Consumer(Dependency[str]):
        broken: str = Depends(Broken)

        async def __aenter__(self) -> str: ...

    def make_consumer() -> str:
        return cast(str, Consumer())

    async def my_func(c: str = make_consumer()) -> None: ...

    async with resolved_dependencies(my_func) as deps:
        assert isinstance(deps["c"], FailedDependency)
        assert isinstance(deps["c"].error, RuntimeError)


def test_accessing_dep_before_aenter_raises() -> None:
    def get_value() -> str: ...

    class MyDep(Dependency[str]):
        value: str = Depends(get_value)

        async def __aenter__(self) -> str: ...

    instance = MyDep()
    with pytest.raises(AttributeError):
        instance.value  # noqa: B018


def test_get_class_dependencies_returns_deps() -> None:
    class Inner(Dependency[int]):
        async def __aenter__(self) -> int: ...

    class MyDep(Dependency[str]):
        x: int = Depends(Inner)

        async def __aenter__(self) -> str: ...

    deps = get_class_dependencies(MyDep)
    assert "x" in deps
    assert isinstance(deps["x"], Dependency)


async def test_standalone_usage_without_resolved_dependencies() -> None:
    def get_value() -> str:
        return "standalone"

    class StandaloneDep(Dependency[str]):
        value: str = Depends(get_value)

        async def __aenter__(self) -> str:
            return self.value

    async with StandaloneDep() as result:
        assert result == "standalone"


def test_unwrap_raises_for_missing_method() -> None:
    class Empty:
        pass

    with pytest.raises(TypeError, match="Empty has no __aenter__"):
        _unwrap(Empty, "__aenter__", "__original__")  # type: ignore[arg-type]
