"""Round-trip migrations in an isolated, disposable database."""

import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from sqlalchemy import create_engine, inspect, text

from app.core.config import Settings


def test_migration_upgrade_downgrade_and_model_consistency():
    database_name = f"taibifer_test_{uuid4().hex}"
    admin = create_engine(Settings().database_url, isolation_level="AUTOCOMMIT")
    # The name is generated here, never derived from user input.
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{database_name}"'))
    environment = {**os.environ, "POSTGRES_DB": database_name}
    database = create_engine(Settings().database_url.set(database=database_name))

    def alembic(*arguments):
        result = subprocess.run(
            [sys.executable, "-m", "alembic", *arguments],
            env=environment, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    try:
        alembic("upgrade", "head")
        alembic("upgrade", "head")  # Repeating an upgrade must be safe.
        assert set(inspect(database).get_table_names()) == {
            "alembic_version", "documents", "document_versions", "document_chunks"
        }
        alembic("check")  # No drift between mapped models and migration.
        alembic("downgrade", "base")
        assert inspect(database).get_table_names() == ["alembic_version"]
        with database.connect() as connection:
            assert connection.scalar(text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")) == 1
        # Generate into a writable temporary directory, never the source tree.
        with TemporaryDirectory() as directory:
            script_directory = Path(directory)
            (script_directory / "versions").mkdir()
            for filename in ("env.py", "script.py.mako"):
                shutil.copy(Path("alembic") / filename, script_directory / filename)
            config_path = script_directory / "alembic.ini"
            config_path.write_text(f"[alembic]\nscript_location = {directory}\n")
            alembic("-c", str(config_path), "revision", "--autogenerate", "-m", "vector import check")
            generated = next((script_directory / "versions").glob("*.py"))
            assert "from pgvector.sqlalchemy import VECTOR" in generated.read_text()
            assert callable(runpy.run_path(str(generated))["upgrade"])
        alembic("upgrade", "head")
        alembic("check")
    finally:
        database.dispose()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{database_name}" WITH (FORCE)'))
        admin.dispose()
