FROM python:3.7

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "setuptools<60" "Cython<3"

COPY requirements_final.txt .

RUN pip install --no-cache-dir --no-build-isolation --no-deps -r requirements_final.txt

RUN pip install google-cloud-storage

RUN python -m spacy download en_core_web_sm

ENV HF_HOME=/app/models

RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('roberta-base'); AutoTokenizer.from_pretrained('roberta-base')"

COPY ./app/model_gector_temp /app/models/finetuned_v10

COPY . /app

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["sh", "-c", "uvicorn gector_infer_api:app --host 0.0.0.0 --port 8080"]