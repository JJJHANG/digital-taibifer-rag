"""Run after installing backend/ into the active Python environment."""

from app.ingestion.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
