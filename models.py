"""
models.py
=========
ORM models for the Urdu OCR Suite: Document (one uploaded PDF) and Page (one
OCR'd page belonging to a document).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship

from database import Base

# Document processing states.
STATUS_PENDING = "pending"        # uploaded, estimate shown, awaiting confirm
STATUS_PROCESSING = "processing"  # OCR run in flight
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    # Editable metadata (dashboard inline-edit targets).
    title = Column(String, nullable=False, default="")
    author = Column(String, nullable=False, default="")
    year = Column(String, nullable=False, default="")  # string: tolerates "1899", "n.d.", etc.

    # File / processing bookkeeping.
    filename = Column(String, nullable=False)
    pdf_path = Column(String, nullable=False)
    num_pages = Column(Integer, nullable=False, default=0)
    pages_ocred = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default=STATUS_PENDING)
    error = Column(Text, nullable=False, default="")

    # Pre-flight token/cost estimate (computed at upload).
    est_input_tokens = Column(Integer, nullable=False, default=0)
    est_output_tokens = Column(Integer, nullable=False, default=0)
    est_cost_usd = Column(Float, nullable=False, default=0.0)

    date_added = Column(DateTime, nullable=False, default=datetime.utcnow)

    pages = relationship(
        "Page",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Page.page_number",
    )

    def avg_confidence(self) -> float:
        done = [p.confidence for p in self.pages if p.text]
        return round(sum(done) / len(done), 3) if done else 0.0

    def verified_count(self) -> int:
        return sum(1 for p in self.pages if p.verified)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "filename": self.filename,
            "num_pages": self.num_pages,
            "pages_ocred": self.pages_ocred,
            "verified_count": self.verified_count(),
            "status": self.status,
            "error": self.error,
            "avg_confidence": self.avg_confidence(),
            "est_input_tokens": self.est_input_tokens,
            "est_output_tokens": self.est_output_tokens,
            "est_cost_usd": round(self.est_cost_usd, 4),
            "date_added": self.date_added.isoformat() if self.date_added else None,
        }


class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False)  # 1-based

    image_path = Column(String, nullable=False, default="")  # rendered page PNG
    text = Column(Text, nullable=False, default="")          # editable reconciled text
    confidence = Column(Float, nullable=False, default=0.0)
    notes = Column(Text, nullable=False, default="")         # reconciliation / model notes
    utrnet_text = Column(Text, nullable=False, default="")
    gemini_text = Column(Text, nullable=False, default="")
    verified = Column(Boolean, nullable=False, default=False)  # human-proofread flag

    document = relationship("Document", back_populates="pages")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
            "verified": bool(self.verified),
        }
