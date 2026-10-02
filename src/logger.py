"""Единая настройка логирования для всего проекта.

Все модули должны получать логгер через `setup_logger(__name__)`, чтобы
в сообщениях было корректное имя модуля.
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "heart_disease",
    level: int = logging.INFO,
) -> logging.Logger:
    """Вернуть сконфигурированный логгер.

    Идемпотентен: повторные вызовы не добавляют дублирующиеся хендлеры.

    Args:
        name: Имя логгера, обычно `__name__` вызывающего модуля.
        level: Уровень логирования (по умолчанию INFO).

    Returns:
        Настроенный экземпляр `logging.Logger`.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
    )
    logger.addHandler(handler)

    # Не передаём сообщения в root logger — избегаем дублей
    logger.propagate = False

    return logger
