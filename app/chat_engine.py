# app/chat_engine.py
"""
对话引擎 - 支持 streaming（SSE）
"""
import uuid
import json
from typing import List, Optional, Dict, AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

from app.config import get_settings
from app.vector_store_manager import VectorStoreManager
from app.prompts import RAG_SYSTEM_PROMPT


# ===== 会话记忆管理 =====
class ConversationMemory:
    """会话记忆"""

    def __init__(self, max_messages: int = 20):
        self._conversations: Dict[str, List] = {}
        self.max_messages = max_messages

    def get_history(self, conversation_id: str) -> List:
        return self._conversations.get(conversation_id, [])

    def add_message(self, conversation_id: str, role: str, content: str):
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []

        if role == "human":
            self._conversations[conversation_id].append(HumanMessage(content=content))
        elif role == "ai":
            self._conversations[conversation_id].append(AIMessage(content=content))
        elif role == "system":
            self._conversations[conversation_id].append(SystemMessage(content=content))

        if len(self._conversations[conversation_id]) > self.max_messages:
            self._conversations[conversation_id] = \
                self._conversations[conversation_id][-self.max_messages:]

    def clear(self, conversation_id: str):
        self._conversations.pop(conversation_id, None)


class ChatEngine:
    """对话引擎"""

    def __init__(self):
        self.settings = get_settings()
        self.vector_store_manager = VectorStoreManager()
        self.memory = ConversationMemory()
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM"""
        # 普通同步 LLM
        self.llm = ChatOpenAI(
            model=self.settings.llm_model,
            openai_api_key=self.settings.openai_api_key,
            openai_api_base=self.settings.openai_api_base,
            temperature=0.3,
            max_tokens=2000,
        )
        # 流式 LLM
        self.llm_streaming = ChatOpenAI(
            model=self.settings.llm_model,
            openai_api_key=self.settings.openai_api_key,
            openai_api_base=self.settings.openai_api_base,
            temperature=0.3,
            max_tokens=2000,
            streaming=True,       # ← 开启流式
        )

    def _build_context(self, kb_name: str, question: str,
                       top_k: int = None,
                       score_threshold: float = None) -> tuple:
        """
        检索相关文档，构建 context 和 sources
        Returns: (context_string, sources_list)
        """
        search_results = self.vector_store_manager.similarity_search(
            kb_name=kb_name,
            query=question,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        if search_results:
            context_parts = []
            sources = []
            for doc, score in search_results:
                source_info = doc.metadata.get("source_filename", "未知来源")
                page = doc.metadata.get("page", "")
                page_info = f" 第{page}页" if page else ""
                context_parts.append(
                    f"【来源: {source_info}{page_info}】\n{doc.page_content}"
                )
                sources.append({
                    "filename": source_info,
                    "page": page,
                    "score": round(score, 4),
                    "content_preview": doc.page_content[:200],
                })
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "（未找到相关文档）"
            sources = []

        return context, sources

    def _build_messages(self, question: str, context: str,
                        conversation_id: str = None) -> list:
        """构建发给 LLM 的完整 messages 列表"""
        system_content = RAG_SYSTEM_PROMPT.format(context=context)
        messages = [SystemMessage(content=system_content)]

        if conversation_id:
            history = self.memory.get_history(conversation_id)
            messages.extend(history)

        messages.append(HumanMessage(content=question))
        return messages

    # ==================== 非流式（保留兼容） ====================

    def chat(
        self,
        question: str,
        knowledge_base: str = "default",
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        conversation_id: Optional[str] = None,
    ) -> dict:
        """RAG 对话（非流式）"""
        context, sources = self._build_context(
            knowledge_base, question, top_k, score_threshold)
        messages = self._build_messages(question, context, conversation_id)
        response = self.llm.invoke(messages)
        answer = response.content

        if conversation_id:
            self.memory.add_message(conversation_id, "human", question)
            self.memory.add_message(conversation_id, "ai", answer)

        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "retrieved_documents": len(sources),
        }

    # ==================== 流式输出（SSE） ====================

    async def chat_stream(
        self,
        question: str,
        knowledge_base: str = "default",
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        RAG 对话（流式）— 使用 Server-Sent Events 格式

        SSE 事件类型:
          - event: sources   → 引用来源（开始时发送一次）
          - event: token     → 每个文本片段
          - event: done      → 流式结束，附带完整信息
          - event: error     → 错误信息
        """
        conv_id = conversation_id or str(uuid.uuid4())

        try:
            # 1. 检索文档
            context, sources = self._build_context(
                knowledge_base, question, top_k, score_threshold)

            # 2. 先发送 sources 事件
            sources_data = json.dumps({
                "sources": sources,
                "conversation_id": conv_id,
                "retrieved_documents": len(sources),
            }, ensure_ascii=False)
            yield f"event: sources\ndata: {sources_data}\n\n"

            # 3. 构建消息
            messages = self._build_messages(question, context, conv_id)

            # 4. 流式调用 LLM
            full_answer = ""
            async for chunk in self.llm_streaming.astream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    # SSE 格式：event: token\ndata: {"content": "..."}\n\n
                    token_data = json.dumps({"content": token}, ensure_ascii=False)
                    yield f"event: token\ndata: {token_data}\n\n"

            # 5. 保存到会话记忆
            self.memory.add_message(conv_id, "human", question)
            self.memory.add_message(conv_id, "ai", full_answer)

            # 6. 发送结束事件
            done_data = json.dumps({
                "conversation_id": conv_id,
                "full_answer": full_answer,
            }, ensure_ascii=False)
            yield f"event: done\ndata: {done_data}\n\n"

        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n"

    def clear_conversation(self, conversation_id: str):
        self.memory.clear(conversation_id)
