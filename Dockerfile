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

RUN pip install "Cython<3"

COPY requirements_final.txt .

RUN pip install --no-cache-dir --no-build-isolation --no-deps -r requirements_final.txt

RUN python -m spacy download en_core_web_sm

COPY . /app

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "gector_infer_api:app", "--host", "0.0.0.0", "--port", "8080"]