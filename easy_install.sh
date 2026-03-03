#!/bin/bash
export VLLM_TAG=0.16.0 # use current tag version vllm
export VLLM_COMMIT=89a77b10846fd96273cce78d86d2556ea582d26e
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-${VLLM_TAG}-cp38-abi3-manylinux_2_31_x86_64.whl
if [ "$1" == "--env-only" ]; then
    echo $VLLM_PRECOMPILED_WHEEL_LOCATION
else
    export SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_TAG}
    {
        cd requirements
        pip install -r common.txt
        pip install -r build.txt
        cd ..
        pip install --editable .[audio]
    }
fi