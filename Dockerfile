FROM python:3.11.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt
COPY budget_tool_router ./budget_tool_router
COPY scripts ./scripts
COPY run_service.py ./
RUN useradd --create-home --uid 10001 toolbandit
USER toolbandit
EXPOSE 8080
CMD ["python", "run_service.py"]
