import pytest
import pytest_asyncio
import requests

from typing import Callable

from ...utils import RemoteOpenAIServer

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
        "--enable-launch-arguments"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="function", params=[False, True])
def server_with_launch_arguments_disabled(
        request,
        monkeypatch_module):

    use_v1 = request.param
    monkeypatch_module.setenv('VLLM_USE_V1', '1' if use_v1 else '0')

    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "100",
        "--enforce-eager",
        "--max-num-seqs",
        "18"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_launch_arguments_with_curl(server: RemoteOpenAIServer):
    url = f"http://localhost:{server.port}/v1/launch_arguments"
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers).json()
    target_response = {"model": MODEL_NAME,
                       "task": "auto",
                       "enable_launch_arguments": True,
                       "max_model_len": 333,
                       "max_num_seqs": 13,
                       "dtype": "bfloat16"}
    for key, value in target_response.items():
        assert response[key] == value

    clean_arguments = {key: value.__name__ if isinstance(value, Callable) else value
                           for key, value in vars(server.args).items()}
    for key, value in clean_arguments.items():
        assert response[key] == value


@pytest.mark.asyncio
async def test_launch_arguments_disabled_with_curl(server_with_launch_arguments_disabled: RemoteOpenAIServer):
    url = f"http://localhost:{server_with_launch_arguments_disabled.port}/v1/launch_arguments"
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers).json()
    target_response = {'object': 'error',
                        'message': 'Launch arguments is not enabled',
                        'type': 'BadRequestError',
                        'code': 400}
    for key, value in target_response.items():
        assert response[key] == value