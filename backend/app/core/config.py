from pydantic_settings import BaseSettings
import os
from pathlib import Path
from typing import Optional, List

# Get the root directory (two levels up from this file)
ROOT_DIR = Path(__file__).parent.parent.parent.parent

class Settings(BaseSettings):
    # API Keys / Models
    openai_api_key: str
    openai_chat_model: str = "gpt-3.5-turbo"
    openai_embedding_model: str = "text-embedding-3-small"

    # Paths & Configs
    litepaper_src_dir: str
    vector_db_dir: str = "./data/vector_store"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 150
    search_k: int = 4

    # CORS Configuration
    cors_origins: str = "http://localhost:3000,http://localhost:3001,http://localhost:3002"

    @property
    def is_github_repo(self) -> bool:
        """Check if the litepaper source is a GitHub repository URL."""
        return self.litepaper_src_dir.startswith(("https://github.com/", "git@github.com:"))

    @property
    def repo_name(self) -> Optional[str]:
        """Extract the repository name from the GitHub URL."""
        if not self.is_github_repo:
            return None
        if self.litepaper_src_dir.endswith(".git"):
            return self.litepaper_src_dir.split("/")[-1][:-4]
        return self.litepaper_src_dir.split("/")[-1]

    class Config:
        env_file = ROOT_DIR / '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings() 