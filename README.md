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

`Depends`, `Shared` and `CallArgument` also work as `Annotated` metadata, so
you can name a dependency once and reuse it:

```python
DB = Annotated[Connection, Depends(get_db)]

async def handle_request(db: DB):
    await db.execute("SELECT 1")
```

This resolves like `db: Connection = Depends(get_db)`, but to Python `db` is
still a required parameter. A custom dependency supplies a value from metadata
only if it subclasses `Provider`; other dependencies there wrap the call.

When something else calls the function with type-checked arguments, such as a
task queue that schedules it, default the parameter to `Provided` so those
callers can leave it out:

```python
async def handle_request(db: DB = Provided): ...
```

A `type` alias works the same way: `type DB = Annotated[Connection, Depends(get_db)]`.

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

When the outer function doesn't declare the parameter,
`without_dependencies` adds it to the wrapper's signature as keyword-only,
required unless the reference has a `default`. Optional references are never
added.

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
