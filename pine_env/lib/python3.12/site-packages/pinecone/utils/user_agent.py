from .version import __version__
from .constants import SOURCE_TAG
import re


def _build_source_tag_field(source_tag):
    # normalize source tag
    # 1. Lowercase
    # 2. Limit charset to [a-z0-9_ :]
    # 3. Trim left/right whitespace
    # 4. Condense multiple spaces to one, and replace with underscore
    tag = source_tag.lower()
    tag = re.sub(r"[^a-z0-9_ :]", "", tag)
    tag = tag.strip()
    tag = "_".join(tag.split())
    return f"{SOURCE_TAG}={tag}"


def _get_user_agent(client_id, config):
    user_agent = (
        f"{client_id}; {_build_source_tag_field(config.source_tag)}"
        if config.source_tag
        else client_id
    )
    return user_agent


def get_user_agent(config):
    return _get_user_agent(f"python-client-{__version__}", config)


def get_user_agent_grpc(config):
    return _get_user_agent(f"python-client[grpc]-{__version__}", config)
