# uncalled-for

Async dependency injection for Python functions.

Declare what your function needs as parameter defaults. They show up resolved
when the function runs. No ceremony, no container, no configuration.

```python
from uncalled_for import Depends

async def get_db():
    db = await connect()
    try:
        yield db
    finally:
        await db.close()

async def handle_request(db: Connection = Depends(get_db)):
    await db.execute("SELECT 1")
```

## Features

- **Zero dependencies** — standard library only
- **Async-native** — built on `AsyncExitStack` and `ContextVar`
- **Context manager lifecycle** — sync and async generators get proper cleanup
- **Nested dependencies** — dependencies can depend on other dependencies
- **Caching** — each dependency resolves once per call, even if referenced multiple times
- **Call arguments** — a factory can reference the arguments of the call it serves

## Annotated dependencies

`Depends`, `Shared` and `CallArgument` also work as `Annotated` metadata, which
lets you name a dependency once and reuse it:

```python
from typing import Annotated

DB = Annotated[Connection, Depends(get_db)]

async def handle_request(db: DB):
    await db.execute("SELECT 1")
```

When the function runs through `resolved_dependencies` or
`without_dependencies`, this resolves exactly like
`db: Connection = Depends(get_db)`: the parameter is reported by
`get_dependency_parameters` and hidden by `without_dependencies`. To Python,
though, `db` is still a required parameter, so calling `handle_request()`
directly is an error. The default form keeps that call legal, which matters to
type checkers and to callers that skip dependency resolution. A parameter may
declare its value with a default or with metadata, not both.

`Depends`, `Shared` and `CallArgument` are `Provider`s. Subclass `Provider`
instead of `Dependency` for a custom dependency that should supply the value
from metadata. Other `Dependency` subclasses in `Annotated` metadata don't
supply a value: they wrap the call and receive the parameter's value through
`bind_to_parameter`.

### Side effects

Before 0.5.0, `Annotated[T, Depends(factory)]` ran the factory around the call
and threw its value away. To keep that behavior, put the work in a
`Dependency` subclass, which runs around the call without supplying a value:

```python
class Audit(Dependency[None]):
    async def __aenter__(self) -> None:
        await audit_log()

async def task(customer_id: Annotated[int, Audit()]):
    ...
```

## Call arguments

A factory sometimes needs a value from the call itself, not from another
dependency. `CallArgument` declares that reference:

```python
from uncalled_for import CallArgument, Depends

def load_user(user_id: str = CallArgument()) -> User:
    return users[user_id]

async def handle_request(
    user_id: str,
    user: User = Depends(load_user),
):
    ...
```

The bare form takes the name of the parameter it is declared on.
`CallArgument("name")` names a different parameter of the outer function. The
reference sees the same value the outer function receives, whether the caller
passed it or another dependency on the outer signature produced it.
`CallArgument(optional=True)` yields `None` when the outer function has no
such argument. References that form a cycle raise `CycleError`.

## Bindings

`Depends` accepts keyword bindings, so you can wire up an ordinary function
without changing it:

```python
async def handle_request(
    user_id: str,
    user: User = Depends(load_user, user_id=CallArgument("user_id")),
):
    ...
```

A binding that is a `Dependency`, such as `CallArgument(...)` or another
`Depends(...)`, resolves first and the factory receives its value. Any other
value passes through as it is. A binding replaces the factory parameter's own
default, which is then never resolved. Two dependencies on the same factory
share one cached result only when their bindings match.

## Install

```
pip install uncalled-for
```
