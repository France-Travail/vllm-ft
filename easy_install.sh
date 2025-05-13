#!/bin/bash

export VLLM_COMMIT=ed2462030f2ccc84be13d8bb2c7476c84930fb71 # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
if [ "$1" == "--docker_build" ]; then
    {
        cd requirements
        pip install -r common.txt
        pip install -r build.txt
        cd ..
        export LATEST_TAG=echo "(git describe --tags `git rev-list --tags --max-count=1`)"
        git checkout $LATEST_TAG
        pip install --editable .
    }
else
    {
        cd requirements
        pip install -r common.txt
        pip install -r build.txt
        cd ..
        pip install --editable .
    }
fi