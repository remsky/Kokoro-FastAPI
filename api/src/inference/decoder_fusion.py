"""Opt-in decoder fusion: freeze weight-norm, fuse Snake via CUDA kernel.

Enable with KOKORO_DECODER_FUSION=1 after the model is on CUDA.
Fails closed when requirements are not met; never enables a CPU fallback.
Validated on Quadro K620 (sm_50): ~8.5% wall-time reduction with exact audio.
"""

from __future__ import annotations

import torch

from . import fused_snake


def enable(model):
    if model.training or model.device.type != "cuda":
        raise RuntimeError("Decoder fusion requires an evaluation model on CUDA")
    if any(
        p.dtype != torch.float32 or p.device != model.device for p in model.parameters()
    ):
        raise RuntimeError("Decoder fusion requires FP32 parameters on one CUDA device")

    with torch.no_grad():
        kernel = fused_snake.Snake()
        frozen = 0
        for layer in model.modules():
            if hasattr(layer, "weight_g") and hasattr(layer, "weight_v"):
                torch.nn.utils.remove_weight_norm(layer)
                frozen += 1
        blocks = fused_snake.install(model, kernel)
    return {"frozen_weights": frozen, "fused_blocks": len(blocks)}
