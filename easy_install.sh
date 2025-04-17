#!/bin/bash

cd requirements
pip install -r common.txt
pip install -r build.txt
cd ..
VLLM_USE_PRECOMPILED=1 pip install --editable .