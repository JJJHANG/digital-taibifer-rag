"""Verify ownership and integrity rules against real PostgreSQL."""

from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.models import Document, DocumentChunk, DocumentVersion
from app.models.documents import EMBEDDING_DIMENSIONS


def make_document(session):
    document = Document(title="生物多樣性指南", source_path=f"raw/{uuid4()}.md", source_type="markdown")
    session.add(document)
    session.flush()
    return document


def make_version(session, document):
    version = DocumentVersion(document_id=document.id, content_hash="a" * 64, version_number=1)
    session.add(version)
    session.flush()
    return version


def make_chunk(version):
    return DocumentChunk(
        document_version_id=version.id,
        chunk_index=0,
        content="臺灣生物多樣性知識",
        chunk_metadata={"page_number": 1, "language": "zh-TW"},
        embedding=[1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1),
    )


def test_document_version_chunk_round_trip(session):
    document = make_document(session)
    version = make_version(session, document)
    assert version.status == "pending"
    assert version.chunking_config == {}
    chunk = make_chunk(version)
    session.add(chunk)
    version.status = "ready"
    document.active_version_id = version.id
    session.flush()
    session.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
    session.expire_all()

    stored = session.scalar(
        select(DocumentChunk)
        .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
        .join(Document, Document.active_version_id == DocumentVersion.id)
        .where(Document.id == document.id)
    )
    assert stored.content == "臺灣生物多樣性知識"
    assert stored.chunk_metadata["page_number"] == 1
    assert len(stored.embedding) == EMBEDDING_DIMENSIONS
    assert stored.created_at.tzinfo is not None
    distance = session.scalar(
        select(DocumentChunk.embedding.cosine_distance(chunk.embedding))
        .where(DocumentChunk.id == stored.id)
    )
    assert distance == pytest.approx(0)


def test_active_version_must_belong_to_document(session):
    first = make_document(session)
    second = make_document(session)
    version = make_version(session, second)
    first.active_version_id = version.id
    # The FK is deferred; flush alone is insufficient to test it.
    session.flush()
    with pytest.raises(IntegrityError):
        session.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))


@pytest.mark.parametrize("duplicate", ["source_path", "version_number", "chunk_index"])
def test_duplicates_are_rejected(session, duplicate):
    document = make_document(session)
    version = make_version(session, document)
    session.add(make_chunk(version))
    session.flush()
    if duplicate == "source_path":
        session.add(Document(title="重複文件", source_path=document.source_path, source_type="markdown"))
    elif duplicate == "version_number":
        session.add(DocumentVersion(document_id=document.id, content_hash="b" * 64, version_number=1))
    else:
        session.add(make_chunk(version))
    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize(
    ("field", "value"),
    [("version_number", 0), ("status", "unknown")],
)
def test_invalid_version_fields_are_rejected(session, field, value):
    document = make_document(session)
    version = make_version(session, document)
    setattr(version, field, value)
    with pytest.raises(IntegrityError):
        session.flush()


def test_negative_chunk_index_is_rejected(session):
    version = make_version(session, make_document(session))
    chunk = make_chunk(version)
    chunk.chunk_index = -1
    session.add(chunk)
    with pytest.raises(IntegrityError):
        session.flush()


def test_wrong_vector_dimensions_are_rejected_by_database(session):
    version = make_version(session, make_document(session))
    # Raw SQL bypasses the Python vector adapter, testing the DB constraint.
    with pytest.raises(DBAPIError):
        session.execute(
            text("INSERT INTO document_chunks (document_version_id, chunk_index, content, embedding) "
                 "VALUES (:version_id, 0, 'invalid', '[1,2,3]'::vector)"),
            {"version_id": version.id},
        )


def test_deleting_document_cascades_versions_and_chunks(session):
    document = make_document(session)
    version = make_version(session, document)
    chunk = make_chunk(version)
    session.add(chunk)
    document.active_version_id = version.id
    session.flush()
    chunk_id, version_id = chunk.id, version.id
    session.execute(delete(Document).where(Document.id == document.id))
    session.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
    assert session.scalar(select(func.count()).select_from(DocumentVersion).where(DocumentVersion.id == version_id)) == 0
    assert session.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.id == chunk_id)) == 0
