import os
import json
import pytest
import pytest_asyncio
import requests

from typing import Callable

from ...utils import RemoteOpenAIServer
from vllm.entrypoints.openai import utils_ft
from vllm.version import ORIGINAL_VLLM_VERSION
from ...conftest import _TEST_DIR

# any model should work
MODEL_NAME = "facebook/opt-125m"

@pytest.fixture(scope="function")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="function", params=[False, True])
def server(
        request,
        monkeypatch_module):

    use_v1 = request.param
    monkeypatch_module.setenv('VLLM_USE_V1', '1' if use_v1 else '0')

    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "333",
        "--enforce-eager",
        "--max-num-seqs",
        "13",
        "--api-endpoint-prefix",
        "/hey"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_info_with_prefix(server: RemoteOpenAIServer):
    url = f"http://localhost:{server.port}/hey/v1/info"
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers).json()
    target_response = {"model_name": MODEL_NAME,
                       "application": "vllm_ft",
                       "version": utils_ft.get_package_version(),
                       "vllm_version": ORIGINAL_VLLM_VERSION,
                       "max_length": 333,
                       "extra_information": {}}
    for key, value in target_response.items():
        assert response[key] == value