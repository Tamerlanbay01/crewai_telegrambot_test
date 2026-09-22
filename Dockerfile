FROM docker:27.3.1-cli AS docker-cli

FROM python:3.12.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/app

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# DockerSkillSandbox talks to the dedicated Docker daemon from Compose.
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

COPY requirements-runtime.txt requirements-telegram.txt requirements-storage.txt requirements-memory.txt requirements-scheduler.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-runtime.txt

COPY app ./app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/.crewai /skill-workspace \
    && chown -R 10001:10001 /app /skill-workspace

USER 10001:10001

CMD ["python", "app/main.py"]
