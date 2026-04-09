from vllm.ft_version import FORK_VERSION


def get_package_version() -> str:
    '''Returns the current version of the package
    Returns:
        str: version of the package
    '''
    return FORK_VERSION.replace("v", "").split("+")[0]