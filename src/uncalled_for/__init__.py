"""Async dependency injection for Python functions.

Declare dependencies as parameter defaults. They resolve automatically when
the function is called through the dependency resolution context manager.
"""

from ._annotations import get_annotation_dependencies
from ._base import Dependency
from ._functional import DependencyFactory, Depends
from ._introspection import get_dependency_parameters, get_signature
from ._resolution import FailedDependency, resolved_dependencies, without_dependencies
from ._shared import Shared, SharedContext
from ._validation import validate_dependencies

__all__ = [
    "Dependency",
    "DependencyFactory",
    "Depends",
    "FailedDependency",
    "Shared",
    "SharedContext",
    "get_annotation_dependencies",
    "get_dependency_parameters",
    "get_signature",
    "resolved_dependencies",
    "validate_dependencies",
    "without_dependencies",
]
