import importlib.metadata

try:
    from vllm.ft_version import FORK_VERSION
except ImportError:
    FORK_VERSION = "unknown"


def get_package_version() -> str:
    '''Returns the current version of the package
    Returns:
        str: version of the package
    '''
    return FORK_VERSION.replace("v", "").split("+")[0]