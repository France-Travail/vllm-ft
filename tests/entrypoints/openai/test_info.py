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
        "13"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server

@pytest.fixture(scope="function", params=[False, True])
def server_with_extra_information(
        request,
        monkeypatch_module):

    use_v1 = request.param
    monkeypatch_module.setenv('VLLM_USE_V1', '1' if use_v1 else '0')
    extra_information_path = str(os.path.join(_TEST_DIR, "entrypoints", "openai", "data", "extra_information.json"))
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "333",
        "--enforce-eager",
        "--max-num-seqs",
        "13",
        "--extra-information",
        f"{extra_information_path}"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_info_with_curl(server: RemoteOpenAIServer):
    url = f"http://localhost:{server.port}/v1/info"
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers).json()
    target_response = {"model_name": MODEL_NAME,
                       "application": "vllm_ft",
                       "version": utils_ft.get_package_version().split("+")[0],
                       "vllm_version": ORIGINAL_VLLM_VERSION,
                       "max_length": 333,
                       "extra_information": {}}
    for key, value in target_response.items():
        assert response[key] == value


@pytest.mark.asyncio
async def test_info_with_extra_information(server_with_extra_information: RemoteOpenAIServer):
    url = f"http://localhost:{server_with_extra_information.port}/v1/info"
    headers = {
        "Content-Type": "application/json",
    }
    extra_information_path = str(os.path.join(_TEST_DIR, "entrypoints", "openai", "data", "extra_information.json"))
    with open(extra_information_path, 'r') as json_file:
        extra_information = json.load(json_file)
    response = requests.get(url, headers=headers).json()
    target_response = {"model_name": MODEL_NAME,
                       "application": "vllm_ft",
                       "version": utils_ft.get_package_version().split("+")[0],
                       "vllm_version": ORIGINAL_VLLM_VERSION,
                       "max_length": 333,
                       "extra_information": extra_information}
    for key, value in target_response.items():
        if key == "extra_information":
            for key_extra_information, value_extra_information in value.items():
                assert response["extra_information"][key_extra_information] == value_extra_information
        else:
            assert response[key] == value