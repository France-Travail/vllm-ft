#!/bin/bash
export VLLM_TAG=$(grep "ORIGINAL_VLLM_VERSION" vllm/version.py | cut -d '"' -f 2)

echo "Version vLLM actuel ${VLLM_TAG}"

WHEEL_NAME="vllm-${VLLM_TAG}-cp38-abi3-manylinux_2_31_x86_64.whl"
echo "Recherche du commit pour la wheel ${WHEEL_NAME}..."
VLLM_COMMIT=$(curl -sL "https://wheels.vllm.ai/${VLLM_TAG}/vllm" | grep -oE "[a-f0-9]{40}/${WHEEL_NAME}" | head -n 1 | cut -d '/' -f 1)
if [ -z "$VLLM_COMMIT" ]; then
    echo "Erreur : Impossible de trouver un commit correspondant pour la wheel ${WHEEL_NAME} sur la page de release."
    exit 1
fi
export VLLM_COMMIT
echo "Commit trouvé : ${VLLM_COMMIT}"
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${VLLM_COMMIT}/${WHEEL_NAME}"

# Check if URL is OK
echo "Lien de récupération de la wheel : ${VLLM_PRECOMPILED_WHEEL_LOCATION}"
HTTP_STATUS=$(curl -o /dev/null -s -I -w "%{http_code}\n" "$VLLM_PRECOMPILED_WHEEL_LOCATION")
if [ "$HTTP_STATUS" -ne 200 ]; then
    echo "Erreur : La wheel n'est pas accessible au lien généré (Erreur HTTP $HTTP_STATUS)"
    exit 1
fi

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