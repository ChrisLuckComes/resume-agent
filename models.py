from datetime import datetime

from pydantic import BaseModel
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    Enum,
)
from sqlalchemy.dialects.postgresql import JSONB
from database import Base
import enum


class AIResponse(BaseModel):
    content: str
    model_name: str


class VectorData(BaseModel):
    text: str
    embedding: List[float]  # 向量，一个浮点数列表


# 定义简历处理状态枚举 📊
class ResumeStatus(enum.Enum):
    PENDING = "pending"  # 等待处理
    PARSING = "parsing"  # 正在解析/向量化
    EVALUATING = "evaluating"  # 正在 AI 评估
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 处理失败


class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), nullable=False)  # 上传者的用户ID
    candidate_name = Column(String(100), index=True)  # 候选人姓名
    phone = Column(String(20), nullable=False)  # 电话号码
    status = Column(Enum(ResumeStatus), default=ResumeStatus.PENDING)  # 处理状态
    content = Column(Text)  # 简历文本内容
    evaluation_result = Column(JSONB, nullable=True)  # AI评估结果，存储为JSON字符串
    created_at = Column(DateTime, default=datetime.now)  # 上传时间戳
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )  # 更新时间戳

    __table_args__ = (
        # 确保同一用户上传的简历中候选人姓名唯一，避免重复上传同一候选人简历
        UniqueConstraint("user_id", "phone", name="_user_phone_uc"),
        Index("ix_user_phone", "user_id", "phone"),
    )


class ResumeEvaluation(BaseModel):
    decision: str  # 决策结果，例如 "通过"、"不通过"、"需要进一步评估"
    match_score: int  # 0-10
    tech_stack: List[str]  # 识别出的关键词
    key_achievements: List[str]  # C（过程）+B（架构）+A（数据） 的实质产出
    risks: List[str]  # 风险点，例如 "缺乏相关经验"、"技能不匹配"、"过于频繁跳槽" 等
    ai_bonus: Optional[str]  # AI 额外加分项，例如 "具备领导力潜质"、"有跨领域经验" 等


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), index=True)
    role = Column(String(50))  # 'user' 或 'model'
    content = Column(Text)  # 消息内容
    created_at = Column(DateTime, default=datetime.now)  # 消息时间戳


class QueryRequest(BaseModel):
    user_id: str  # 标识用户
    text: str


class EvaluationRequest(BaseModel):
    candidate_name: str  # 候选人姓名
    user_id: str  # 用户ID
    phone: str  # 手机号码
