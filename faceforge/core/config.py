import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FACEFORGE_")

    # Project root
    project_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # Model configuration
    confidence_threshold: float = Field(default=0.40)
    nms_threshold: float = Field(default=0.40)
    input_size: tuple[int, int] = Field(default=(640, 640))

    # Server configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)

    # Database
    database_url: str = Field(default="postgresql+asyncpg://postgres:secret@localhost:5432/postgres")

    # Logging
    log_level: str = Field(default="INFO")

    @property
    def model_path(self) -> Path:
        return Path(__file__).parent.parent / "models" / "scrfd_10g_bnkps.onnx"

    @property
    def images_dir(self) -> Path:
        return Path(__file__).parent.parent / "images"

    @property
    def output_dir(self) -> Path:
        return Path(__file__).parent.parent / "output"


settings = Settings()