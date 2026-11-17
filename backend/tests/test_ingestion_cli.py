"""Exercise the CLI with synthetic manifests, mocked HTTP, and real PostgreSQL."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI
import pytest
from sqlalchemy import create_engine, func, select
import yaml

from app.core.config import EmbeddingSettings
from app.ingestion import cli
from app.ingestion.embedding import OpenAIEmbeddingProvider
from app.models import Document, DocumentChunk, DocumentVersion


def make_manifest(tmp_path, names=("guide",), *, deferred=True):
    root = tmp_path / "knowledge"
    (root / "retrieval").mkdir(parents=True)
    entries = []
    for name in names:
        (root / "retrieval" / f"{name}.md").write_text(f"# {name}\nSynthetic evidence for {name}.\n")
        entries.append({
            "id": name, "source_path": f"{tmp_path.name}/{name}.md", "title": name,
            "source_type": "markdown", "path": f"retrieval/{name}.md", "ingestion_status": "prepared",
        })
    if deferred:
        entries.append({"id": "deferred-csv", "source_path": f"{tmp_path.name}/terms.csv", "title": "Terms",
                        "source_type": "csv", "path": None, "ingestion_status": "deferred_csv_strategy"})
    path = root / "manifest.yaml"
    data = {"schema_version": 1, "source_root": "/not-mounted-originals",
            "path_base": "manifest_directory", "documents": entries}
    path.write_text(yaml.safe_dump(data))
    return path, data


def rewrite(path, data):
    path.write_text(yaml.safe_dump(data))


def test_dry_run_needs_no_key_or_database(tmp_path, monkeypatch, capsys):
    path, _ = make_manifest(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not construct an SDK client or DB engine")

    monkeypatch.setattr(cli, "OpenAIEmbeddingProvider", forbidden)
    monkeypatch.setattr(cli, "create_async_engine", forbidden)
    assert cli.main(["--manifest", str(path), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "dry_run"
    assert report["totals"]["selected_documents"] == 1
    assert report["totals"]["estimated_tokens_before_dedup"] > 0
    assert report["totals"]["usage"]["requests"] == 0
    assert report["excluded"][0]["reason"] == "deferred_csv_strategy"
    assert "Synthetic evidence" not in json.dumps(report)


@pytest.mark.parametrize("selector", ["id", "relative_file", "absolute_file"])
def test_selection_by_id_or_prepared_path(tmp_path, capsys, selector):
    path, _ = make_manifest(tmp_path, ("first", "second"))
    choices = {"id": ["--document-id", "second"],
               "relative_file": ["--file", "retrieval/second.md"],
               "absolute_file": ["--file", str(path.parent / "retrieval/second.md")]}
    assert cli.main(["--manifest", str(path), "--dry-run", *choices[selector]]) == 0
    assert [row["id"] for row in json.loads(capsys.readouterr().out)["documents"]] == ["second"]


@pytest.mark.parametrize("fault", [
    "duplicate_id", "duplicate_source", "duplicate_path", "traversal", "absolute",
    "evaluation_path", "symlink", "missing", "csv_prepared", "faq", "null_prepared",
])
def test_unsafe_or_inconsistent_manifests_fail_before_api(tmp_path, capsys, monkeypatch, fault):
    path, data = make_manifest(tmp_path, ("first", "second"))
    entry = data["documents"][1]
    if fault.startswith("duplicate_"):
        key = {"duplicate_id": "id", "duplicate_source": "source_path", "duplicate_path": "path"}[fault]
        entry[key] = data["documents"][0][key]
    elif fault == "traversal":
        entry["path"] = "retrieval/../evaluation/answer.md"
    elif fault == "absolute":
        entry["path"] = str(path.parent / "retrieval/second.md")
    elif fault == "evaluation_path":
        entry["path"] = "evaluation/answer.md"
    elif fault == "symlink":
        (path.parent / "answer.md").write_text("Private evaluation answer")
        (path.parent / "retrieval/second.md").unlink()
        (path.parent / "retrieval/second.md").symlink_to(path.parent / "answer.md")
    elif fault == "missing":
        (path.parent / "retrieval/second.md").unlink()
    elif fault == "csv_prepared":
        entry["source_type"] = "csv"
    elif fault == "faq":
        (path.parent / "retrieval/second.md").write_text("# Body\nEvidence\n# FAQ 問答集\n**Q: Held out?**\nA: Secret\n")
    elif fault == "null_prepared":
        entry["path"] = None
    rewrite(path, data)
    calls = []
    monkeypatch.setattr(cli, "OpenAIEmbeddingProvider", lambda *args: calls.append(args))
    assert cli.main(["--manifest", str(path)]) == 2
    assert calls == []
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert "Secret" not in json.dumps(report)


@pytest.mark.parametrize("selection", [
    ["--document-id", "missing-id"], ["--document-id", "deferred-csv"],
    ["--file", "evaluation/faq/answers.md"],
])
def test_invalid_explicit_selection_fails(tmp_path, capsys, selection):
    path, _ = make_manifest(tmp_path)
    assert cli.main(["--manifest", str(path), "--dry-run", *selection]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def mock_provider(monkeypatch, requests, database_url, *, fail_on=None):
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if fail_on and fail_on in " ".join(payload["input"]):
            return httpx.Response(500, json={"error": {"message": "Do not expose this API body", "type": "server_error"}})
        return httpx.Response(200, json={
            "object": "list", "model": payload["model"],
            "data": [{"object": "embedding", "index": i, "embedding": [1.0] + [0.0] * 1535}
                     for i in range(len(payload["input"]))],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        })

    def build(settings):
        client = AsyncOpenAI(api_key="synthetic-key", max_retries=0, base_url="https://example.invalid/v1",
                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        return OpenAIEmbeddingProvider(settings, client=client)

    monkeypatch.setattr(cli, "OpenAIEmbeddingProvider", build)
    monkeypatch.setattr(cli, "Settings", lambda: SimpleNamespace(database_url=database_url))


def test_cli_persists_and_second_run_is_unchanged(tmp_path, monkeypatch, capsys, index_database_url):
    path, data = make_manifest(tmp_path)
    requests = []
    mock_provider(monkeypatch, requests, index_database_url)
    assert cli.main(["--manifest", str(path)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["totals"]["indexed"] == 1
    assert cli.main(["--manifest", str(path)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["totals"]["unchanged"] == 1 and second["totals"]["usage"]["requests"] == 0
    assert len(requests) == 1
    assert first["documents"][0]["version_id"] == second["documents"][0]["version_id"]
    engine = create_engine(index_database_url)
    try:
        with engine.connect() as connection:
            document_id = connection.scalar(select(Document.id).where(Document.source_path == data["documents"][0]["source_path"]))
            assert connection.scalar(select(func.count()).select_from(DocumentVersion).where(DocumentVersion.document_id == document_id)) == 1
            assert connection.scalar(select(func.count()).select_from(DocumentChunk).join(
                DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id
            ).where(DocumentVersion.document_id == document_id)) == 1
    finally:
        engine.dispose()


@pytest.mark.parametrize("keep_going", [False, True])
def test_failed_document_exit_status_and_batch_policy(tmp_path, monkeypatch, capsys, index_database_url, keep_going):
    path, _ = make_manifest(tmp_path, ("first", "fail", "last"))
    requests = []
    mock_provider(monkeypatch, requests, index_database_url, fail_on="fail")
    options = ["--continue-on-error"] if keep_going else []
    assert cli.main(["--manifest", str(path), *options]) == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert [row["status"] for row in report["documents"]] == [
        "indexed", "failed", "indexed" if keep_going else "not_attempted"
    ]
    assert "Do not expose this API body" not in output.out + output.err
