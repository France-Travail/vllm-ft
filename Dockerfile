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

RUN pip install --upgrade pip

WORKDIR /app

RUN python -m venv Venv_vllm_ft
RUN source /app/Venv_vllm_ft/bin/activate
ENV VIRTUAL_ENV="/app/Venv_vllm_ft" PATH="/app/Venv_vllm_ft/bin:${PATH}"

# Install package
COPY pyproject.toml setup.py README.md easy_install.sh /app/
COPY requirements /app/requirements/
COPY vllm /app/vllm

RUN chmod +x /app/easy_install.sh

WORKDIR /app/requirements
RUN pip install -r common.txt
RUN pip install -r build.txt

WORKDIR /app
RUN --mount=type=bind,source=.git,target=/app/.git \
    export LATEST_TAG=echo "(git describe --tags `git rev-list --tags --max-count=1`)" \
    && git checkout $LATEST_TAG \
    && export VLLM_PRECOMPILED_WHEEL_LOCATION=$(/app/easy_install.sh --env-only) \
    && VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_PRECOMPILED_WHEEL_LOCATION pip install .

# Start API
EXPOSE 5000
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]
