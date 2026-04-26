FROM python:3.11-slim

WORKDIR /workspace

RUN pip install --no-cache-dir \
    pydantic>=2.6 \
    pyyaml>=6.0 \
    httpx>=0.27 \
    numpy>=1.26 \
    scipy>=1.11 \
    pytest>=8.0

COPY judge /workspace/judge

ENV PYTHONPATH=/workspace
CMD ["python", "-m", "judge.judge"]
