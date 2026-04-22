# app/config.py
import os
import re
import json
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    llm_model: str = "glm-4-flash"
    embedding_model: str = "embedding-3"
    vectorstore_dir: str = str(BASE_DIR / "data" / "vectorstore")
    documents_dir: str = str(BASE_DIR / "data" / "documents")
    chunk_size: int = 500
    chunk_overlap: int = 50
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    max_upload_size: int = 50 * 1024 * 1024
    top_k: int = 5
    score_threshold: float = 0.7

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"


def safe_dir_name(name: str) -> str:
    """
    将知识库名称转换为安全的 ASCII 目录名
    "面试技巧" → "mian_shi_ji_qiao_8a3f"
    "tech_docs" → "tech_docs"
    """
    # 如果已经是纯 ASCII，直接用
    if name.isascii():
        slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        slug = re.sub(r'_+', '_', slug).strip('_')
        return slug or "default"

    # 中文/日文/韩文等 → 拼音首字母 + 哈希后缀
    import hashlib
    h = hashlib.md5(name.encode('utf-8')).hexdigest()[:6]
    slug = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', '_', name)
    slug = re.sub(r'_+', '_', slug).strip('_')
    # 对非 ASCII 字符做简单转换：用 Unicode 码点映射
    ascii_parts = []
    for ch in slug:
        if ord(ch) > 127:
            ascii_parts.append(f"{ord(ch):x}")
        else:
            ascii_parts.append(ch)
    result = ''.join(ascii_parts)
    return result[:50] if result else f"kb_{h}"


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    Path(settings.vectorstore_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.documents_dir).mkdir(parents=True, exist_ok=True)
    return settings
