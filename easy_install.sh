#!/bin/bash
export VLLM_TAG=$(grep "ORIGINAL_VLLM_VERSION" vllm/version.py | cut -d '"' -f 2)

echo "vLLM's version : ${VLLM_TAG}"


WHEEL_NAME="vllm-${VLLM_TAG}-cp38-abi3-manylinux_2_31_x86_64.whl"

export VLLM_PRECOMPILED_WHEEL_LOCATION="https://github.com/vllm-project/vllm/releases/download/v${VLLM_TAG}/${WHEEL_NAME}"

# Check if URL is OK
echo "Wheel's url : ${VLLM_PRECOMPILED_WHEEL_LOCATION}"
HTTP_STATUS=$(curl -L -o -v /dev/null -s -I -w "%{http_code}\n" "$VLLM_PRECOMPILED_WHEEL_LOCATION")
if [ "$HTTP_STATUS" -ne 200 ]; then
    echo "Error: wheel's url not recheable (HTTP's error $HTTP_STATUS)"
    exit 1
fi


REQ_FILE="common.txt"
if [ "$1" != "--env-only" ] && [ -n "$1" ]; then
    REQ_FILE="$1"
fi

if [ "$1" == "--env-only" ]; then
    echo $VLLM_PRECOMPILED_WHEEL_LOCATION
else
    {
        export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_TAG}"
        export VLLM_VERSION="${VLLM_TAG}"
        cd requirements
        echo "Installation des dépendances depuis ${REQ_FILE}..."
        pip install -r "$REQ_FILE"
        pip install -r build.txt
        cd ..
        pip install --editable .[audio]
    }
fi