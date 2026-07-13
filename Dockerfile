FROM nvidia/cuda:13.3.0-devel-ubuntu22.04 AS builder

LABEL org.opencontainers.image.author="Agence Data Services"
LABEL org.opencontainers.image.description="REST service vllm-ft"

COPY prebuildfs /
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install python common
RUN install_packages software-properties-common git curl ffmpeg

RUN add-apt-repository -d -y 'ppa:deadsnakes/ppa' \
     && install_packages python3.11 python3.11-dev python3.11-venv python3-pip gcc-12 g++-12\
     && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1\
     && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 110 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=true

RUN python -m venv /opt/venv \
    && pip install --upgrade pip
ENV VIRTUAL_ENV="/opt/venv" PATH="/opt/venv/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# Install package
COPY pyproject.toml setup.py README.md easy_install.sh /app/
COPY requirements /app/requirements/
COPY vllm /app/vllm
COPY tools /app/tools
COPY rust /app/rust


RUN chmod +x /app/easy_install.sh

ARG REQ_FILE="common.txt"

RUN --mount=type=bind,source=.git,target=/app/.git \
    /app/easy_install.sh ${REQ_FILE}

# Start API
EXPOSE 5000
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]