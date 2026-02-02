from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    tencent_cloud_secret_id: str = ""
    tencent_cloud_secret_key: str = ""
    tencent_cloud_region: str = ""
    chroma_db_path: str = "./chroma_db"
    rag_images_path: str = "./rag_images"
    dashscope_api_key: str = ""
    model_name: str = ""
    model_provider: str = ""
    api_key: str = ""

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
