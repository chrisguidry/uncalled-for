"""Async dependency injection for Python functions.

Declare dependencies as parameter defaults. They resolve automatically when
the function is called through the dependency resolution context manager.
"""

from .annotations import get_annotation_dependencies
from .base import Dependency, Provided, Provider
from .frames import CallArgument, CycleError, Frame, current_frame, frame_scope
from .functional import DependencyFactory, Depends
from .introspection import (
    get_dependency_parameters,
    get_signature,
)
from .references import get_call_argument_parameters
from .resolution import FailedDependency, resolved_dependencies, without_dependencies
from .shared import Shared, SharedContext
from .validation import validate_dependencies

__all__ = [
    "CallArgument",
    "CycleError",
    "Dependency",
    "DependencyFactory",
    "Depends",
    "FailedDependency",
    "Provided",
    "Provider",
    "Frame",
    "Shared",
    "SharedContext",
    "current_frame",
    "frame_scope",
    "get_annotation_dependencies",
    "get_call_argument_parameters",
    "get_dependency_parameters",
    "get_signature",
    "resolved_dependencies",
    "validate_dependencies",
    "without_dependencies",
]
