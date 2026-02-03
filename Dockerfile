FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY .env.example /app/.env.example
COPY README.md /app/README.md

ENV PYTHONPATH=/app/src
CMD ["python", "-m", "tradebot.app", "run", "--paper"]
