"""Tests for dependencies declared through PEP 695 ``type`` aliases.

conftest.py skips this module on Python 3.11, which cannot parse a ``type``
statement.
"""

from typing import Annotated

from uncalled_for import (
    Dependency,
    Depends,
    Provided,
    get_annotation_dependencies,
    get_dependency_parameters,
    resolved_dependencies,
)


def get_db() -> str:
    return "db"


type DB = Annotated[str, Depends(get_db)]
type GenericDB[T] = Annotated[T, Depends(get_db)]
type AliasOfDB = DB


class Wrapper(Dependency["Wrapper"]):
    async def __aenter__(self) -> "Wrapper": ...


type Wrapped = Annotated[int, Wrapper()]


async def plain_alias(db: DB = Provided) -> None: ...


async def generic_alias(db: GenericDB[str] = Provided) -> None: ...


async def alias_of_alias(db: AliasOfDB = Provided) -> None: ...


async def test_plain_alias_supplies_the_value() -> None:
    async with resolved_dependencies(plain_alias) as arguments:
        assert arguments == {"db": "db"}


async def test_generic_alias_supplies_the_value() -> None:
    async with resolved_dependencies(generic_alias) as arguments:
        assert arguments == {"db": "db"}


async def test_alias_of_an_alias_supplies_the_value() -> None:
    async with resolved_dependencies(alias_of_alias) as arguments:
        assert arguments == {"db": "db"}


def test_alias_holding_a_wrapper_wraps_the_call() -> None:
    async def task(value: Wrapped) -> None: ...

    assert get_dependency_parameters(task) == {}
    assert [type(d) for d in get_annotation_dependencies(task)["value"]] == [Wrapper]


def test_metadata_around_an_alias_adds_to_the_aliases_metadata() -> None:
    async def task(db: Annotated[DB, Wrapper()] = Provided) -> None: ...

    assert set(get_dependency_parameters(task)) == {"db"}
    assert [type(d) for d in get_annotation_dependencies(task)["db"]] == [Wrapper]
