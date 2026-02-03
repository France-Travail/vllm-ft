#!/bin/bash
export VLLM_TAG=0.15.0 # use current tag version vllm
export VLLM_COMMIT=f176443446f659dbab5315e056e605d8984fd976
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