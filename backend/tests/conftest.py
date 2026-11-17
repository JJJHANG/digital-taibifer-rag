"""Integration tests use PostgreSQL; each test rolls its data back."""

import os
import subprocess
import sys
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core.config import Settings


@pytest.fixture
def session():
    engine = create_engine(Settings().database_url)
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            with Session(bind=connection) as session:
                yield session
            if transaction.is_active:
                transaction.rollback()
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def index_database_url():
    """Allow real commits without touching the development corpus."""
    name = f"taibifer_index_test_{uuid4().hex}"
    url = Settings().database_url
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            env={**os.environ, "POSTGRES_DB": name}, capture_output=True, text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        yield url.set(database=name)
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()
