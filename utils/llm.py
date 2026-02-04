import sys
import os
import time
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

# Add the project root to sys.path to allow imports from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


def create_llm_with_retry():
    """Create LLM instance with retry logic for better reliability"""
    return ChatOpenAI(
        # DeepSeek 模型名（必填，根据需求选）
        model=settings.model_name,
        # DeepSeek API Key（从 .env 读取，或直接填写）
        api_key=SecretStr(settings.api_key),
        # DeepSeek 兼容 OpenAI 的 API 地址（固定）
        base_url="https://api.deepseek.com",
        # 温度（越低越精准，适合总结场景）
        temperature=0,
        # 超时时间 - 增加到120秒以避免超时
        timeout=300,
        # 添加重试配置
        max_retries=3,
    )


llm = create_llm_with_retry()
