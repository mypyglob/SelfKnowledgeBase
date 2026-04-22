# app/models.py
"""
Pydantic 数据模型定义
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChatMessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ===== 请求模型 =====
class ChatRequest(BaseModel):
    """聊天请求"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    knowledge_base: str = Field("default", description="知识库名称")
    top_k: Optional[int] = Field(None, description="检索文档数量")
    score_threshold: Optional[float] = Field(None, description="相关性阈值")
    conversation_id: Optional[str] = Field(None, description="会话ID")


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50,
                       pattern=r'^[a-zA-Z0-9_\u4e00-\u9fff\-]+$',
                       description="知识库名称")
    description: Optional[str] = Field("", max_length=200)


class KnowledgeBaseUpdate(BaseModel):
    description: Optional[str] = Field(None, max_length=200)


# ===== 响应模型 =====
class ChatResponse(BaseModel):
    """非流式聊天响应"""
    answer: str = Field(..., description="AI 回答")
    sources: List[dict] = Field(default_factory=list, description="引用来源")
    conversation_id: Optional[str] = Field(None)


class KnowledgeBaseResponse(BaseModel):
    name: str
    description: str
    document_count: int
    created_at: str


class APIResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: Optional[dict | list] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    knowledge_bases: int = 0
