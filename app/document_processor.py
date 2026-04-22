"""
文档加载与处理模块
支持: PDF, TXT, Markdown, DOCX, CSV, JSON
"""
import os
import json
import csv
import chardet
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Generator, Optional

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from app.config import get_settings


# ===== 支持的文件类型映射 =====
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "markdown",
    ".docx": "docx",
    ".csv": "csv",
    ".json": "json",
}


class DocumentProcessor:
    """文档处理器"""

    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def detect_encoding(self, file_path: str) -> str:
        """检测文件编码"""
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)  # 读取前 10000 字节
            result = chardet.detect(raw_data)
        return result.get("encoding", "utf-8")

    def load_document(self, file_path: str) -> List[Document]:
        """
        根据文件类型加载文档
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            # 为每个文档添加页码元数据
            for i, doc in enumerate(docs):
                doc.metadata["page"] = i + 1

        elif ext == ".txt":
            encoding = self.detect_encoding(file_path)
            loader = TextLoader(file_path, encoding=encoding)
            docs = loader.load()

        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()

        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()

        elif ext == ".csv":
            loader = CSVLoader(file_path, encoding=self.detect_encoding(file_path))
            docs = loader.load()

        elif ext == ".json":
            docs = self._load_json(file_path)

        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        # 统一添加文件名到元数据
        filename = Path(file_path).name
        for doc in docs:
            doc.metadata["source_filename"] = filename
            doc.metadata["file_type"] = ext

        return docs

    def _load_json(self, file_path: str) -> List[Document]:
        """加载 JSON 文件"""
        encoding = self.detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding) as f:
            data = json.load(f)

        docs = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                content = json.dumps(item, ensure_ascii=False, indent=2)
                docs.append(Document(
                    page_content=content,
                    metadata={"index": i, "source_filename": Path(file_path).name, "file_type": ".json"}
                ))
        elif isinstance(data, dict):
            content = json.dumps(data, ensure_ascii=False, indent=2)
            docs.append(Document(
                page_content=content,
                metadata={"source_filename": Path(file_path).name, "file_type": ".json"}
            ))
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档切分为小块（chunks）
        """
        # 对 Markdown 文件使用特殊分割器保留标题结构
        md_docs = []
        other_docs = []

        for doc in documents:
            if doc.metadata.get("file_type") == ".md":
                md_docs.append(doc)
            else:
                other_docs.append(doc)

        chunks = []

        # Markdown 使用标题分割
        if md_docs:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
            )
            for doc in md_docs:
                md_chunks = markdown_splitter.split_text(doc.page_content)
                # 合并元数据
                for chunk in md_chunks:
                    chunk.metadata.update(doc.metadata)
                # 如果 Markdown 块太大，继续用 RecursiveCharacterTextSplitter
                md_chunks = self.text_splitter.split_documents(md_chunks)
                chunks.extend(md_chunks)

        # 其他文件使用通用分割器
        if other_docs:
            other_chunks = self.text_splitter.split_documents(other_docs)
            chunks.extend(other_chunks)

        return chunks

    def process_file(self, file_path: str) -> Tuple[List[Document], dict]:
        """
        完整处理流程：加载 → 切分 → 返回
        Returns: (文档块列表, 处理信息字典)
        """
        start_time = datetime.now()

        # 1. 加载文档
        documents = self.load_document(file_path)

        # 2. 过滤空文档
        documents = [doc for doc in documents if doc.page_content.strip()]

        # 3. 切分文档
        chunks = self.split_documents(documents)

        processing_time = (datetime.now() - start_time).total_seconds()

        info = {
            "filename": Path(file_path).name,
            "original_chunks": len(documents),
            "final_chunks": len(chunks),
            "processing_time": round(processing_time, 2),
            "file_size": os.path.getsize(file_path),
        }

        return chunks, info
