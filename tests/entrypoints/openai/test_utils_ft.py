import importlib.metadata

from vllm.entrypoints.openai import utils_ft

def test_get_package_version():
    # Nominal case
    version = utils_ft.get_package_version()
    assert version == importlib.metadata.version("vllm")