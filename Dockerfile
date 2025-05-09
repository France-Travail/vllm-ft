FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

LABEL org.opencontainers.image.author="Agence Data Services"
LABEL org.opencontainers.image.description="REST service vllm-ft"

COPY prebuildfs /
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install python common
RUN install_packages software-properties-common git

RUN add-apt-repository -d -y 'ppa:deadsnakes/ppa' \
     && install_packages python3.11 python3.11-dev python3.11-venv python3-pip gcc-10 g++-10\
     && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1\
     && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=true

RUN python -m venv /opt/venv \
    && pip install --upgrade pip
ENV VIRTUAL_ENV="/opt/venv" PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Install package
COPY pyproject.toml setup.py README.md easy_install.sh LICENSE MANIFEST.in /app/
COPY requirements /app/requirements/
COPY vllm /app/vllm

RUN chmod +x /app/easy_install.sh
WORKDIR /app/requirements
RUN pip install -r common.txt
RUN pip install -r build.txt
WORKDIR /app
ENV VLLM_COMMIT=ed2462030f2ccc84be13d8bb2c7476c84930fb71
ENV VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/ed2462030f2ccc84be13d8bb2c7476c84930fb71/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
ENV CUDA_VERSION=12.8.1
# pip install .

RUN --mount=type=bind,source=.git,target=/app/.git \
    python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
RUN pip install dist/*.whl --extra-index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# Start API
EXPOSE 5000
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]
