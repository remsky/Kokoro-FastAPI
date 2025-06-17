from typing import List, Tuple, Any, Dict


def parse_non_empty_args(args: List[Tuple[str, Any]]) -> Dict[str, Any]:
    return {arg_name: val for arg_name, val in args if val is not None}
