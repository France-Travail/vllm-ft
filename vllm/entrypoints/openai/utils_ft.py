import importlib.metadata

def get_package_version() -> str:
    '''Returns the current version of the package
    Returns:
        str: version of the package
    '''
    version = importlib.metadata.version("vllm")
    return version