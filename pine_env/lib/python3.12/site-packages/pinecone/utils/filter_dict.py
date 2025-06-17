from typing import Tuple, Dict


def filter_dict(d: Dict, allowed_keys: Tuple[str, ...]) -> Dict:
    return {k: v for k, v in d.items() if k in allowed_keys}
