from datetime import datetime
import enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Enum, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class RadarMetric(BaseModel):
    name: str
    value: int = Field(..., ge=0, le=100)
    max: int = Field(default=100, ge=1)


class EvaluationResult(BaseModel):
    summary: str
    title: str
    decision: str
    match_score: int = Field(..., ge=0, le=100)
    radar_metrics: List[RadarMetric] = Field(default_factory=list)
    highlights: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class ResumeStatus(enum.Enum):
    PENDING = "pending"
    PARSING = "parsing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(50), nullable=False)
    candidate_name: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    phone: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[ResumeStatus] = mapped_column(
        Enum(ResumeStatus), default=ResumeStatus.PENDING, nullable=False
    )
    content: Mapped[Optional[str]] = mapped_column(Text)
    evaluation_result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("user_id", "phone", name="_user_phone_uc"),
        Index("ix_user_phone", "user_id", "phone"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class QueryRequest(BaseModel):
    user_id: str
    text: str
    candidate_name: Optional[str] = None
    resume_id: Optional[int] = None


class EvaluationRequest(BaseModel):
    user_id: str
    jd_text: str
    resume_id: Optional[int] = None
    candidate_name: Optional[str] = None
    phone: Optional[str] = None
    target_seniority: Optional[str] = None
    jd_keywords: Optional[List[str]] = None


class JDAnalysisRequest(BaseModel):
    jd_text: str
    target_seniority: Optional[str] = None


class JDAnalysisResponse(BaseModel):
    keywords: List[str] = Field(default_factory=list)


class JDKeywordExtractionResult(BaseModel):
    keywords: List[str] = Field(default_factory=list, description="仅包含来自JD原文的关键词")


class ChatRequest(BaseModel):
    user_id: str
    text: str
    role: str = "user"
    candidate_name: Optional[str] = None
    resume_id: Optional[int] = None


class OCRResponse(BaseModel):
    text: str


class ChatSuggestionsResponse(BaseModel):
    suggestions: List[str] = Field(default_factory=list)
