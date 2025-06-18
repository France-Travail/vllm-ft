#!/bin/bash

export VLLM_TAG=0.9.1 # use current tag version vllm
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_TAG}/vllm-${VLLM_TAG}-cp38-abi3-manylinux1_x86_64.whl
if [ "$1" == "--env-only" ]; then
    echo $VLLM_PRECOMPILED_WHEEL_LOCATION
else
    {
        cd requirements
        pip install -r common.txt
        pip install -r build.txt
        cd ..
        pip install --editable .
    }
fi