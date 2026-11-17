"""Request-scoped sessions; callers explicitly own transaction boundaries."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Close the session (and roll back unfinished work) after each request.

    Use ``async with session.begin():`` around a unit of work to commit it.
    A session must not be shared between concurrent asyncio tasks.
    """
    async with request.app.state.session_factory() as session:
        yield session
