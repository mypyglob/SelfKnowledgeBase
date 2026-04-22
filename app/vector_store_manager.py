# app/vector_store_manager.py
"""
向量库管理模块 - 修复 FAISS 中文路径问题
使用 ASCII 安全目录名 + 名称映射文件
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from app.config import get_settings, safe_dir_name


class VectorStoreManager:
    """向量库管理器（单例模式）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.settings = get_settings()
        self._stores: dict = {}
        self._embeddings = None
        self._name_map: dict = {}    # {display_name: dir_name}
        self._dir_map: dict = {}     # {dir_name: display_name}
        self._init_embeddings()
        self._load_name_map()

    def _init_embeddings(self):
        """初始化 Embedding 模型"""
        self._embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key,
            openai_api_base=self.settings.openai_api_base,
        )

    # ====================== 名称映射管理 ======================

    def _get_map_path(self) -> Path:
        """获取名称映射文件路径"""
        return Path(self.settings.vectorstore_dir) / "_name_map.json"

    def _load_name_map(self):
        """启动时加载名称映射"""
        map_path = self._get_map_path()
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._name_map = data.get("name_to_dir", {})
                self._dir_map = data.get("dir_to_name", {})

    def _save_name_map(self):
        """保存名称映射"""
        map_path = self._get_map_path()
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump({
                "name_to_dir": self._name_map,
                "dir_to_name": self._dir_map,
            }, f, ensure_ascii=False, indent=2)

    def _kb_to_dir(self, kb_name: str) -> str:
        """知识库名称 → 安全目录名"""
        if kb_name in self._name_map:
            return self._name_map[kb_name]
        dir_name = safe_dir_name(kb_name)
        self._name_map[kb_name] = dir_name
        self._dir_map[dir_name] = kb_name
        self._save_name_map()
        return dir_name

    def _dir_to_kb(self, dir_name: str) -> str:
        """安全目录名 → 知识库显示名称"""
        return self._dir_map.get(dir_name, dir_name)

    def _get_kb_path(self, kb_name: str) -> Path:
        """获取知识库的实际存储路径（ASCII安全）"""
        dir_name = self._kb_to_dir(kb_name)
        return Path(self.settings.vectorstore_dir) / dir_name

    def _get_kb_meta_path(self, kb_name: str) -> Path:
        """获取知识库元数据路径"""
        return self._get_kb_path(kb_name) / "meta.json"

    # ====================== 知识库 CRUD ======================

    def list_knowledge_bases(self) -> List[dict]:
        """列出所有知识库"""
        kb_dir = Path(self.settings.vectorstore_dir)
        if not kb_dir.exists():
            return []

        knowledge_bases = []
        for item in kb_dir.iterdir():
            if item.is_dir() and item.name.startswith("_"):
                continue  # 跳过映射文件
            if item.is_dir() and (item / "index.faiss").exists():
                # 通过目录名反查显示名称
                display_name = self._dir_to_kb(item.name)
                meta = self._load_kb_meta(display_name)
                knowledge_bases.append({
                    "name": display_name,
                    "description": meta.get("description", ""),
                    "document_count": meta.get("document_count", 0),
                    "created_at": meta.get("created_at", ""),
                    "total_chunks": meta.get("total_chunks", 0),
                })
        return knowledge_bases

    def _load_kb_meta(self, kb_name: str) -> dict:
        meta_path = self._get_kb_meta_path(kb_name)
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_kb_meta(self, kb_name: str, meta: dict):
        meta_path = self._get_kb_meta_path(kb_name)
        # ✅ 关键修复：确保父目录存在
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def create_knowledge_base(self, kb_name: str, description: str = "") -> dict:
        """创建新知识库"""
        kb_path = self._get_kb_path(kb_name)
        if kb_path.exists() and (kb_path / "index.faiss").exists():
            raise ValueError(f"知识库 '{kb_name}' 已存在")

        # ✅ 关键修复：先用 os.makedirs 创建目录（比 Path.mkdir 更可靠）
        os.makedirs(str(kb_path), exist_ok=True)

        # 用空文档初始化 FAISS 索引
        dummy_doc = Document(page_content="init", metadata={"_dummy": True})
        vectorstore = FAISS.from_documents([dummy_doc], self._embeddings)
        vectorstore.save_local(str(kb_path))

        # 保存元数据
        meta = {
            "name": kb_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "document_count": 0,
            "total_chunks": 0,
            "files": [],
        }
        self._save_kb_meta(kb_name, meta)

        # 保存名称映射
        self._save_name_map()

        # 缓存
        self._stores[kb_name] = vectorstore

        return meta

    def delete_knowledge_base(self, kb_name: str) -> bool:
        """删除知识库"""
        kb_path = self._get_kb_path(kb_name)
        if not kb_path.exists():
            return False

        if kb_name in self._stores:
            del self._stores[kb_name]

        shutil.rmtree(kb_path)

        # 清理名称映射
        dir_name = self._kb_to_dir(kb_name)
        if dir_name in self._dir_map:
            del self._dir_map[dir_name]
        if kb_name in self._name_map:
            del self._name_map[kb_name]
        self._save_name_map()

        return True


    def _load_or_get_store(self, kb_name: str) -> FAISS:
        """加载或获取缓存的向量库（兼容新旧版本）"""
        if kb_name in self._stores:
            return self._stores[kb_name]

        kb_path = self._get_kb_path(kb_name)
        if not kb_path.exists() or not (kb_path / "index.faiss").exists():
            raise ValueError(f"知识库 '{kb_name}' 不存在")

        # ✅ 兼容性加载：自动判断是否支持 allow_dangerous_deserialization 参数
        import inspect
        load_params = inspect.signature(FAISS.load_local).parameters

        if "allow_dangerous_deserialization" in load_params:
            # 新版本 langchain-community (>=0.0.10)
            vectorstore = FAISS.load_local(
                str(kb_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            # 旧版本 langchain-community
            vectorstore = FAISS.load_local(
                str(kb_path),
                self._embeddings,
            )

        self._stores[kb_name] = vectorstore
        return vectorstore

    def add_documents(
        self, kb_name: str, documents: List[Document], filename: str = ""
    ) -> dict:
        """向知识库添加文档"""
        documents = [doc for doc in documents if not doc.metadata.get("_dummy")]
        if not documents:
            return {"added_chunks": 0, "message": "没有有效文档可添加"}

        vectorstore = self._load_or_get_store(kb_name)
        vectorstore.add_documents(documents)

        kb_path = self._get_kb_path(kb_name)
        # ✅ 关键修复：保存前确保目录存在
        os.makedirs(str(kb_path), exist_ok=True)
        vectorstore.save_local(str(kb_path))

        # 更新元数据
        meta = self._load_kb_meta(kb_name)
        meta["total_chunks"] = meta.get("total_chunks", 0) + len(documents)
        meta["document_count"] = meta.get("document_count", 0) + 1
        if filename and filename not in meta.get("files", []):
            meta.setdefault("files", []).append(filename)
        meta["updated_at"] = datetime.now().isoformat()
        self._save_kb_meta(kb_name, meta)

        return {
            "added_chunks": len(documents),
            "total_chunks": meta["total_chunks"],
        }

    def similarity_search(
        self,
        kb_name: str,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
    ) -> List[Tuple[Document, float]]:
        """相似度搜索"""
        top_k = top_k or self.settings.top_k
        score_threshold = score_threshold or self.settings.score_threshold

        vectorstore = self._load_or_get_store(kb_name)
        results = vectorstore.similarity_search_with_score(query, k=top_k)

        filtered = []
        for doc, score in results:
            if doc.metadata.get("_dummy"):
                continue
            similarity = max(0, 1 - score / 10)
            if similarity >= score_threshold:
                filtered.append((doc, similarity))

        return filtered

    def get_kb_stats(self, kb_name: str) -> dict:
        return self._load_kb_meta(kb_name)
