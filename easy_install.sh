#!/bin/bash
export VLLM_TAG=$(grep "ORIGINAL_VLLM_VERSION" vllm/version.py | cut -d '"' -f 2)

echo "vLLM's version : ${VLLM_TAG}"

WHEEL_NAME="vllm-${VLLM_TAG}-cp38-abi3-manylinux_2_31_x86_64.whl"
echo "Searching for the wheel's commit ${WHEEL_NAME}..."
VLLM_COMMIT=$(curl -sL "https://wheels.vllm.ai/${VLLM_TAG}/vllm" | grep -oE "[a-f0-9]{40}/${WHEEL_NAME}" | head -n 1 | cut -d '/' -f 1)
if [ -z "$VLLM_COMMIT" ]; then
    echo "Error: Could not find a matching commit for wheel ${WHEEL_NAME} on the release page."
    exit 1
fi
export VLLM_COMMIT
echo "Commit found : ${VLLM_COMMIT}"
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${VLLM_COMMIT}/${WHEEL_NAME}"

# Check if URL is OK
echo "Wheel's url : ${VLLM_PRECOMPILED_WHEEL_LOCATION}"
HTTP_STATUS=$(curl -o /dev/null -s -I -w "%{http_code}\n" "$VLLM_PRECOMPILED_WHEEL_LOCATION")
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
        cd requirements
        echo "Installation des dépendances depuis ${REQ_FILE}..."
        pip install -r "$REQ_FILE"
        pip install -r build.txt
        cd ..
        pip install --editable .[audio]
    }
fi