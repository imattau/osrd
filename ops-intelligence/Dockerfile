FROM python:3.12-slim

ARG INSTALL_GNN=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN mkdir -p /var/lib/osrd-ops-intelligence && chown 10001:10001 /var/lib/osrd-ops-intelligence
ENV OSRD_OPS_AUDIT_DB=/var/lib/osrd-ops-intelligence/plans.sqlite3
ENV PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir uv==0.11.29
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN if [ "$INSTALL_GNN" = "true" ]; then \
      uv sync --frozen --no-dev --no-editable --extra gnn; \
    else \
      uv sync --frozen --no-dev --no-editable; \
    fi

USER 10001:10001
EXPOSE 8095
CMD ["uvicorn", "osrd_ops_intelligence.service.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8095"]
