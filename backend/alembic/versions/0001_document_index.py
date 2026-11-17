"""Create document identities, versions, and vector chunks.

Revision ID: 0001
Revises: None
"""

from alembic import op
from pgvector.sqlalchemy import VECTOR
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(20), nullable=False),
        sa.Column("source_url", sa.Text()),
        sa.Column("active_version_id", sa.Uuid()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("source_path", name="uq_documents_source_path"),
        sa.CheckConstraint("source_type IN ('pdf', 'markdown', 'txt')", name=op.f("ck_documents_source_type")),
    )
    op.create_table(
        "document_versions",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column("embedding_model", sa.String(150)),
        sa.Column("chunking_config", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_document_versions"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE", name="fk_document_versions_document_id_documents"),
        sa.UniqueConstraint("document_id", "version_number", name="uq_document_versions_document_id"),
        sa.UniqueConstraint("id", "document_id", name="uq_document_versions_id"),
        sa.CheckConstraint("version_number > 0", name=op.f("ck_document_versions_positive_version_number")),
        sa.CheckConstraint("status IN ('pending', 'processing', 'ready', 'failed')", name=op.f("ck_document_versions_status")),
    )
    op.create_foreign_key(
        "fk_active_document_version", "documents", "document_versions",
        ["active_version_id", "id"], ["id", "document_id"],
        deferrable=True, initially="DEFERRED",
    )
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("document_version_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("embedding", VECTOR(1536), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_document_chunks"),
        sa.ForeignKeyConstraint(["document_version_id"], ["document_versions.id"], ondelete="CASCADE", name="fk_document_chunks_document_version_id_document_versions"),
        sa.UniqueConstraint("document_version_id", "chunk_index", name="uq_document_chunks_document_version_id"),
        sa.CheckConstraint("chunk_index >= 0", name=op.f("ck_document_chunks_nonnegative_chunk_index")),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_document_chunks_document_version_id", "document_chunks", ["document_version_id"])
    op.create_index("ix_document_chunks_metadata", "document_chunks", ["metadata"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_constraint("fk_active_document_version", "documents", type_="foreignkey")
    op.drop_table("document_versions")
    op.drop_table("documents")
    # vector may be used by other schemas; never remove a shared extension.
