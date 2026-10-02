FROM python:3.11-slim

# Не создавать .pyc-файлы + unbuffered stdout (для логов)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Системные зависимости для сборки scikit-learn/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем Poetry
RUN pip install --no-cache-dir poetry==2.5.1

# Копируем только манифесты — чтобы кэш слоёв работал
COPY pyproject.toml poetry.lock ./

# Устанавливаем зависимости (без самого пакета)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main

# Копируем код
COPY src/ ./src/

# Создаём папки для артефактов
RUN mkdir -p models plots results

# Метаданные
LABEL maintainer="Сидоров Павел, Сидорова Екатерина"
LABEL description="Heart Disease Prediction ML Pipeline"
LABEL version="1.0"

# Точка входа
CMD ["python", "-m", "src.automl_pipeline"]
