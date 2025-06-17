from typing import Optional


def normalize_host(host: Optional[str]) -> str:
    if host is None:
        return ""
    if host.startswith("https://"):
        return host
    if host.startswith("http://"):
        return host
    return "https://" + host
