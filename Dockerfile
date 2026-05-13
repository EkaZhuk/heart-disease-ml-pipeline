FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/

RUN mkdir -p models plots results

EXPOSE 5000

LABEL maintainer="Ваша Фамилия Имя"
LABEL description="Heart Disease Prediction ML Pipeline"
LABEL version="1.0"

CMD ["python", "src/automl_pipeline.py"]