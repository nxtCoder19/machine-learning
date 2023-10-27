#FROM --platform=linux/amd64 python:3.11-slim-buster
FROM python:3.11
RUN alias python=python3
RUN pip install -U pip
RUN pip install pdm -U
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY pdm.lock pdm.lock
# RUN pdm install
# COPY . /app
# CMD ["./.venv/bin/python3", "stable-diffusion.py"]