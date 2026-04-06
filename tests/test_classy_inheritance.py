"""Tests for class-level dependency inheritance across Dependency hierarchies."""

from __future__ import annotations

from typing import cast

from uncalled_for import (
    Dependency,
    Depends,
    resolved_dependencies,
)


async def test_nested_class_deps() -> None:
    class Inner(Dependency[str]):
        async def __aenter__(self) -> str:
            return "inner-value"

    class Middle(Dependency[str]):
        inner: str = Depends(Inner)

        async def __aenter__(self) -> str:
            return f"middle({self.inner})"

    class Outer(Dependency[str]):
        middle: str = Depends(Middle)

        async def __aenter__(self) -> str:
            return f"outer({self.middle})"

    async with Outer() as result:
        assert result == "outer(middle(inner-value))"


async def test_inheritance_abstract_parent_with_deps() -> None:
    class Inner(Dependency[int]):
        async def __aenter__(self) -> int:
            return 99

    class AbstractParent(Dependency[str]):
        inner: int = Depends(Inner)

    class ConcreteChild(AbstractParent):
        async def __aenter__(self) -> str:
            return f"child-{self.inner}"

    async with ConcreteChild() as result:
        assert result == "child-99"


async def test_inheritance_both_parent_and_child_have_deps() -> None:
    class DepA(Dependency[str]):
        async def __aenter__(self) -> str:
            return "a"

    class DepB(Dependency[int]):
        async def __aenter__(self) -> int:
            return 2

    class Parent(Dependency[str]):
        a: str = Depends(DepA)

        async def __aenter__(self) -> str: ...

    class Child(Parent):
        b: int = Depends(DepB)

        async def __aenter__(self) -> str:
            return f"child({self.a},{self.b})"

    async with Child() as result:
        assert result == "child(a,2)"


async def test_child_inherits_deps_without_overriding_aenter() -> None:
    class DepA(Dependency[int]):
        async def __aenter__(self) -> int:
            return 1

    class DepB(Dependency[str]):
        async def __aenter__(self) -> str:
            return "b"

    class Parent(Dependency[str]):
        a: int = Depends(DepA)

        async def __aenter__(self) -> str:
            return f"parent({self.a})"

    class Child(Parent):
        b: str = Depends(DepB)

    async with Child() as result:
        assert result == "parent(1)"


async def test_plain_subclass_inherits_parent_wrapper() -> None:
    class Inner(Dependency[int]):
        async def __aenter__(self) -> int:
            return 7

    class Parent(Dependency[str]):
        x: int = Depends(Inner)

        async def __aenter__(self) -> str:
            return f"val={self.x}"

    class Child(Parent):
        pass

    async with Child() as result:
        assert result == "val=7"


async def test_shared_factory_across_class_hierarchy() -> None:
    call_count = 0

    def get_pool() -> str:
        nonlocal call_count
        call_count += 1
        return "the-pool"

    class Parent(Dependency[str]):
        pool: str = Depends(get_pool)

        async def __aenter__(self) -> str:
            return f"parent({self.pool})"

    class Child(Parent):
        pool: str = Depends(get_pool)

        async def __aenter__(self) -> str:
            return f"child({self.pool})"

    def make_parent() -> str:
        return cast(str, Parent())

    def make_child() -> str:
        return cast(str, Child())

    async def my_func(
        p: str = make_parent(),
        c: str = make_child(),
    ) -> None: ...

    async with resolved_dependencies(my_func) as resolved:
        assert resolved["p"] == "parent(the-pool)"
        assert resolved["c"] == "child(the-pool)"
        assert call_count == 1


async def test_sibling_class_dependencies_share_factory() -> None:
    call_count = 0

    def get_connection() -> str:
        nonlocal call_count
        call_count += 1
        return "conn"

    class MyDep(Dependency[str]):
        primary: str = Depends(get_connection)
        replica: str = Depends(get_connection)

        async def __aenter__(self) -> str:
            return f"{self.primary},{self.replica}"

    def make_my_dep() -> str:
        return cast(str, MyDep())

    async def my_func(v: str = make_my_dep()) -> None: ...

    async with resolved_dependencies(my_func) as resolved:
        assert resolved["v"] == "conn,conn"
        assert call_count == 1
