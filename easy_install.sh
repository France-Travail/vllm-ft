#!/bin/bash
export VLLM_TAG=0.14.1 # use current tag version vllm
export VLLM_COMMIT=d7de043d55d1dd629554467e23874097e1c48993
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-${VLLM_TAG}-cp38-abi3-manylinux_2_31_x86_64.whl
if [ "$1" == "--env-only" ]; then
    echo $VLLM_PRECOMPILED_WHEEL_LOCATION
else
    {
        cd requirements
        pip install -r common.txt
        pip install -r build.txt
        cd ..
        pip install --editable .[audio]
    }
fi