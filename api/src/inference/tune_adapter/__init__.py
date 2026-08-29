"""Kokoro-82M zero-shot voice tune adapter. See residual.install and enroll.enroll."""

from .enroll import enroll, load, read
from .prosody import rate, resample, stats
from .residual import (
    ROWS,
    STYLE,
    ResFC,
    attach,
    hook_model,
    install,
    load_adapter,
    r_dim_of,
    use,
)
