# Stage 1 : build
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS builder

COPY prebuildfs /
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install package
## Remove software-properties-common because don't use deadsnake in ubuntu22.04 if python [3.10, 3.12]
RUN install_packages software-properties-common git

RUN add-apt-repository -d -y 'ppa:deadsnakes/ppa' \
     && install_packages python3.11 python3.11-dev python3.11-venv python3-pip gcc-10 g++-10\
     && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1\
     && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10 \
     && python -m venv /opt/venv \
     && /opt/venv/bin/pip install --upgrade pip

# Set env variable
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=true

WORKDIR /app

# Copy files
COPY pyproject.toml setup.py README.md easy_install.sh /app/
COPY requirements /app/requirements/
COPY vllm /app/vllm

# Run easy_install.sh
RUN chmod +x /app/easy_install.sh
RUN --mount=type=bind,source=.git,target=/app/.git \ 
    /app/easy_install.sh

# Stage 2 : final image
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS runtime

LABEL org.opencontainers.image.author="Agence Data Services"
LABEL org.opencontainers.image.description="REST service vllm-ft"

# Install python
## Keep git because easy_install.sh run pip install --editable and use setuptools.scm
COPY prebuildfs /
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN install_packages \
    python3.11 git \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Copy from stage 1
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

EXPOSE 5000
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]