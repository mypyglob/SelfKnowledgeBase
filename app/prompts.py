"""
提示词模板管理
"""
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# ===== RAG 系统提示词 =====
RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请基于以下检索到的参考文档来回答用户的问题。

## 回答规则：
1. **优先使用参考文档**：答案应主要基于提供的参考文档内容
2. **标注来源**：在回答的关键信息后标注引用来源，如 [来源:文件名.pdf 第3页]
3. **诚实回答**：如果参考文档中没有相关信息，请明确告知用户，不要编造答案
4. **结构清晰**：使用清晰的格式组织回答，适当使用列表、加粗等
5. **语言一致**：使用与用户问题相同的语言回答
6. **简洁精准**：直接回答问题，避免不必要的冗余信息

## 参考文档：
{context}

## 注意：
- 如果多个文档对同一问题有不同说法，请综合分析并指出差异
- 如果用户的问题超出知识库范围，可以适当补充通用知识，但需明确标注不属于知识库内容
"""

# ===== 有来源引用的 QA 模板 =====
RAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""基于以下参考文档回答问题。如果文档中没有相关信息，请如实告知。

参考文档：
{context}

用户问题：{question}

请回答：""",
    input_variables=["context", "question"],
)

# ===== 多轮对话模板 =====
CONVERSATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# ===== 文档摘要模板 =====
SUMMARY_PROMPT = PromptTemplate(
    template="""请为以下文档生成一个简洁的摘要（100字以内）：

文档内容：
{content}

摘要：""",
    input_variables=["content"],
)
