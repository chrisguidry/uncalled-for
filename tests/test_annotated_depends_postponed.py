"""Tests for Annotated dependencies under postponed annotation evaluation."""

from __future__ import annotations

from typing import Annotated

from uncalled_for import (
    Depends,
    get_annotation_dependencies,
    get_dependency_parameters,
    resolved_dependencies,
    without_dependencies,
)


def get_db() -> str:
    return "db"


DB = Annotated[str, Depends(get_db)]


async def get_user(email: str, db: DB) -> str:
    return f"{email} via {db}"


def test_string_annotations_find_annotated_depends() -> None:
    assert set(get_dependency_parameters(get_user)) == {"db"}
    assert get_annotation_dependencies(get_user) == {}


async def test_string_annotations_resolve_annotated_depends() -> None:
    async with resolved_dependencies(get_user, {"email": "a"}) as arguments:
        assert arguments == {"db": "db"}

    assert await without_dependencies(get_user)(email="a") == "a via db"
