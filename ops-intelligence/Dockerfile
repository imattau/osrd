FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

USER 10001:10001
EXPOSE 8095
CMD ["uvicorn", "osrd_ops_intelligence.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8095"]
