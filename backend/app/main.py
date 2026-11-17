"""ASGI application and process health endpoint."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.v1.chat import router as chat_router
from app.api.v1.search import router as search_router
from app.core.config import ChatSettings, EmbeddingSettings, Settings
from app.ingestion.embedding import OpenAIEmbeddingProvider
from app.rag.generator import OpenAIChatProvider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the connection pool for this application's lifetime."""
    engine = create_async_engine(Settings().database_url, pool_pre_ping=True)
    app.state.session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with AsyncExitStack() as resources:
            settings = EmbeddingSettings()
            provider = None
            if settings.openai_api_key and settings.openai_api_key.get_secret_value().strip():
                provider = await resources.enter_async_context(OpenAIEmbeddingProvider(settings))
            app.state.embedding_provider = provider
            chat_settings = ChatSettings()
            chat_provider = None
            if chat_settings.openai_api_key and chat_settings.openai_api_key.get_secret_value().strip():
                chat_provider = await resources.enter_async_context(OpenAIChatProvider(chat_settings))
            app.state.chat_provider = chat_provider
            yield
    finally:
        await engine.dispose()


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


app = FastAPI(title="Digital TaiBIFer RAG", version="0.1.0", lifespan=lifespan)
app.include_router(search_router)
app.include_router(chat_router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Report API process health; this does not check database readiness."""
    return HealthResponse()
