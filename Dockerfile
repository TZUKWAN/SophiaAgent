FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY sophia/ sophia/
COPY cli.py run_web.py config.yaml ./

RUN pip install --no-cache-dir -e ".[all]"

RUN mkdir -p /root/SophiaWorkspace

EXPOSE 8080

ENV SOPHIA_WORKSPACE=/root/SophiaWorkspace

CMD ["python", "run_web.py", "--host", "0.0.0.0", "--port", "8080"]
