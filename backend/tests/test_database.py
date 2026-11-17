"""Exercise async session cleanup and safe URL construction."""

import asyncio
from contextlib import aclosing
from types import SimpleNamespace

from sqlalchemy import select, text

from app.core.config import Settings
from app.core.database import get_session
from app.main import app, lifespan
from app.models import Document


def test_url_preserves_special_characters_in_credentials():
    settings = Settings(postgres_password="p@ss:/%word", postgres_user="test@example")
    assert settings.database_url.password == "p@ss:/%word"
    assert settings.database_url.username == "test@example"
    assert "p@ss" not in str(settings.database_url)


def test_async_request_session_rolls_back_uncommitted_work():
    async def scenario():
        async with lifespan(app):
            request = SimpleNamespace(app=app)
            async with aclosing(get_session(request)) as sessions:
                session = await anext(sessions)
                assert await session.scalar(text("SELECT 1")) == 1
                document = Document(title="rollback", source_path="test/rollback.md", source_type="markdown")
                session.add(document)
                await session.flush()
                document_id = document.id
            async with app.state.session_factory() as session:
                assert await session.scalar(select(Document).where(Document.id == document_id)) is None

    asyncio.run(scenario())
