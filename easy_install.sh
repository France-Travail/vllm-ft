#!/bin/bash

cd requirements
pip install -r common.txt
pip install -r build.txt
cd ..
export VLLM_COMMIT=ed2462030f2ccc84be13d8bb2c7476c84930fb71 # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
export CUDA_VERSION=12.8.1
# pip install .
python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
pip install --system dist/*.whl --extra-index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')