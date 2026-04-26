FROM python:3.11-slim
COPY docker/stub_ollama.py /app/stub_ollama.py
WORKDIR /app
EXPOSE 11434
HEALTHCHECK --interval=5s --timeout=2s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=1)" || exit 1
CMD ["python", "stub_ollama.py"]
