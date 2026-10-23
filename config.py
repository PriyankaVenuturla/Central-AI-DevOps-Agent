"""Central-Agent Configuration Module

Loads and validates settings for AWS Bedrock, ChromaDB, GitHub,
and the triage pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


# Bedrock bearer token is loaded from .env or environment variable
@dataclass(frozen=True)
class Settings:
    # AWS Bedrock
    bedrock_bearer_token: str = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
    aws_region: str = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "ap-south-1"))
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", "moonshotai.kimi-k2.5")


    # GitHub
    github_token: str = os.getenv("GITHUB_TOKEN", os.getenv("AGENT_GITHUB_TOKEN", ""))
    github_api_url: str = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    webhook_secret: str = os.getenv("GITHUB_WEBHOOK_SECRET", "")

    # ChromaDB & Embeddings
    chroma_persist_dir: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(Path(__file__).parent / "chroma_data")
    )
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # Pipeline Tuning
    max_boundary_lines: int = int(os.getenv("MAX_BOUNDARY_LINES", "50"))
    max_source_files: int = int(os.getenv("MAX_SOURCE_FILES", "5"))
    max_file_chars: int = int(os.getenv("MAX_FILE_CHARS", "15000"))

    # Webhook Server
    server_host: str = os.getenv("WEBHOOK_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("WEBHOOK_PORT", "8000"))


# Initialize singleton instance
settings = Settings()

# Ensure the Bedrock bearer token is also injected into the environment
# for boto3 / Bedrock clients that resolve credentials from os.environ
if settings.bedrock_bearer_token and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = settings.bedrock_bearer_token
