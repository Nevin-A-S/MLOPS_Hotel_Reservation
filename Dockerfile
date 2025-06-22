FROM python:slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends  \
    libgomp1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
    python3 -m pip install --upgrade pip

COPY . .

RUN pip install --no-cache-dir -e .

RUN python pipeline/training_pipeline.py 

EXPOSE 8080

CMD [ "python", "appication.py" ]