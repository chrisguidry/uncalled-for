"""Call-scoped resolution frames."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from typing import Any, cast

from .base import Dependency
from .introspection import get_dependency_parameters, get_signature

_stack: ContextVar[AsyncExitStack] = ContextVar("uncalled_for_stack")


class CycleError(ValueError):
    """Raised when call-argument references form a cycle."""


_current_frame: ContextVar["Frame"] = ContextVar("uncalled_for_frame")


class Frame:
    """Call-scoped state for one resolution.

    A frame maps the outer function's parameter names to the values the caller
    supplied and to the dependencies declared on its signature. It memoizes
    each name once it resolves, so two references to the same parameter get
    the same value.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        provided: dict[str, Any],
        parameters: dict[str, Dependency[Any]],
    ) -> None:
        self.function = function
        self.provided = provided
        self.parameters = parameters
        self.resolving: list[str] = []
        self.resolved: dict[str, Any] = {}
        self.failures: dict[str, Exception] = {}

    async def resolve(self, name: str, *, provided_only: bool = False) -> Any:
        """Produce the value the outer function receives for the parameter *name*.

        Set *provided_only* to limit the lookup to caller-supplied arguments.
        Nothing is resolved and no dependency is entered in that mode.
        """
        if name in self.resolving:
            chain = " -> ".join([*self.resolving, name])
            raise CycleError(
                f"Circular argument reference in {self.function.__qualname__}: {chain}"
            )

        if name in self.provided:
            return self.provided[name]

        if provided_only:
            raise LookupError(
                f"{self.function.__qualname__} received no argument named {name!r}"
            )

        if name in self.failures:
            raise self.failures[name]

        if name in self.resolved:
            return self.resolved[name]

        if name not in self.parameters:
            parameter = get_signature(self.function).parameters.get(name)
            if parameter is None:
                raise LookupError(
                    f"{self.function.__qualname__} has no parameter named {name!r}"
                )
            if parameter.default is inspect.Parameter.empty:
                raise LookupError(
                    f"{self.function.__qualname__} received no value for "
                    f"parameter {name!r}"
                )
            return parameter.default

        dependency = self.parameters[name].for_parameter(name)
        self.resolving.append(name)
        try:
            value = await _stack.get().enter_async_context(dependency)
        except Exception as error:
            self.failures[name] = error
            raise
        finally:
            self.resolving.pop()

        self.resolved[name] = value
        return value


def current_frame() -> Frame:
    """Return the frame of the resolution in progress."""
    try:
        return _current_frame.get()
    except LookupError:
        raise RuntimeError(
            "A resolution frame is required here. Run this inside "
            "frame_scope() or resolved_dependencies()."
        ) from None


@contextmanager
def frame_scope(
    function: Callable[..., Any],
    provided: dict[str, Any] | None = None,
) -> Iterator[Frame]:
    """Make a frame for one call of *function* and set it as the current frame.

    *provided* holds the arguments the caller passed, which take precedence
    over the dependencies on the signature.
    """
    frame = Frame(function, provided or {}, get_dependency_parameters(function))
    token = _current_frame.set(frame)
    try:
        yield frame
    finally:
        _current_frame.reset(token)


class _CallArgument(Dependency[Any]):
    """Dependency on one argument of the outer function."""

    parameter: str | None
    optional: bool

    def __init__(self, parameter: str | None = None, optional: bool = False) -> None:
        self.parameter = parameter
        self.optional = optional

    def for_parameter(self, name: str) -> Dependency[Any]:
        if self.parameter is not None:
            return self
        return type(self)(name, self.optional)

    async def __aenter__(self) -> Any:
        if self.parameter is None:
            raise RuntimeError(
                "A bare CallArgument was never bound to a parameter name. "
                "Declare it as the default of a parameter."
            )

        frame = current_frame()
        try:
            return await frame.resolve(self.parameter)
        except LookupError:
            if self.optional:
                return None
            raise


def CallArgument(parameter: str | None = None, optional: bool = False) -> Any:
    """Declare a dependency on an argument of the outer function.

    The value is whatever the outer function receives for *parameter* on this
    call. That covers an argument the caller passed and a value another
    dependency on the outer signature produced.

    Leave *parameter* empty to use the name of the parameter this is declared
    on. Set *optional* to True to get None when the outer function has no such
    argument. References that form a cycle raise ``CycleError``, which
    *optional* does not suppress.

    Do not use this inside a ``Shared`` factory. A shared value outlives the
    call that built it, so it must not read that call's arguments.

    Example::

        def get_account(user_id: str = CallArgument()) -> Account:
            return accounts[user_id]

        async def show(user_id: str, account: Account = Depends(get_account)):
            ...
    """
    return cast(Any, _CallArgument(parameter, optional))
