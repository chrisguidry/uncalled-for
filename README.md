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

## Install

```
pip install uncalled-for
```
