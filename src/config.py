"""Конфигурация ML-пайплайна.

Значения по умолчанию можно переопределить через переменные окружения
с префиксом `HD_`, например: `HD_RANDOM_STATE=123`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Корневой конфиг проекта: пути, данные, обучение."""

    model_config = SettingsConfigDict(
        env_prefix="HD_",
        extra="ignore",
    )

    # Пути
    data_path: Path = Path("data/heart.csv")
    models_dir: Path = Path("models")
    plots_dir: Path = Path("plots")
    results_dir: Path = Path("results")

    # Данные
    target_column: str = "target"
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    random_state: int = 42

    # Обучение
    cv_folds: int = Field(5, ge=2)

    def ensure_dirs(self) -> None:
        """Создать папки для артефактов, если их нет."""
        for d in (self.models_dir, self.plots_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


config = Config()
