"""CPU-based PyTorch inference backend."""

import gc
from typing import Optional

import numpy as np
import torch
from loguru import logger

from ..builds.models import build_model
from ..core import paths
from ..core.model_config import model_config
from .base import BaseModelBackend


@torch.no_grad()
def forward(model: torch.nn.Module, tokens: list[int], ref_s: torch.Tensor, speed: float) -> np.ndarray:
    """Forward pass through model with memory management.
    
    Args:
        model: PyTorch model
        tokens: Input tokens
        ref_s: Reference signal
        speed: Speed multiplier
        
    Returns:
        Generated audio
    """
    device = ref_s.device
    pred_aln_trg = None
    asr = None
    
    try:
        # Initial tensor setup
        tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        # Split reference signals
        s_content = ref_s[:, 128:].clone().to(device)
        s_ref = ref_s[:, :128].clone().to(device)

        # BERT and encoder pass
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        # Predictor forward pass
        d = model.predictor.text_encoder(d_en, s_content, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)

        # Duration prediction
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()
        del duration, x  # Free large intermediates

        # Alignment matrix construction
        pred_aln_trg = torch.zeros(
            input_lengths.item(),
            pred_dur.sum().item(),
            device=device
        )
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + pred_dur[0, i].item()] = 1
            c_frame += pred_dur[0, i].item()
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        # Matrix multiplications with cleanup
        en = d.transpose(-1, -2) @ pred_aln_trg
        del d
        
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s_content)
        del en

        # Final text encoding and decoding
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        del t_en

        # Generate output
        output = model.decoder(asr, F0_pred, N_pred, s_ref)
        output=output.squeeze().cpu().numpy()
        pred_dur=pred_dur[0].cpu().numpy() 
        return output,pred_dur * (len(output) / sum(pred_dur))
        
    finally:
        # Clean up largest tensors if they were created
        if pred_aln_trg is not None:
            del pred_aln_trg
        if asr is not None:
            del asr


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create attention mask from lengths.
    
    Args:
        lengths: Sequence lengths
        
    Returns:
        Boolean mask tensor
    """
    max_len = lengths.max()
    mask = torch.arange(max_len, device=lengths.device)[None, :].expand(
        lengths.shape[0], -1
    )
    if lengths.dtype != mask.dtype:
        mask = mask.to(dtype=lengths.dtype)
    return mask + 1 > lengths[:, None]


class PyTorchCPUBackend(BaseModelBackend):
    """PyTorch CPU inference backend."""

    def __init__(self):
        """Initialize CPU backend."""
        super().__init__()
        self._device = "cpu"
        self._model: Optional[torch.nn.Module] = None

        # Configure PyTorch CPU settings
        config = model_config.pytorch_cpu
        if config.num_threads > 0:
            torch.set_num_threads(config.num_threads)
        if config.pin_memory:
            torch.set_default_tensor_type(torch.FloatTensor)

    async def load_model(self, path: str) -> None:
        """Load PyTorch model.
        
        Args:
            path: Path to model file
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            
            logger.info(f"Loading PyTorch model on CPU: {model_path}")
            self._model = await build_model(model_path, self._device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")

    def generate(
        self,
        tokens: list[int],
        voice: torch.Tensor,
        speed: float = 1.0
    ) -> np.ndarray:
        """Generate audio using CPU model.
        
        Args:
            tokens: Input token IDs
            voice: Voice embedding tensor
            speed: Speed multiplier
            
        Returns:
            Generated audio samples
            
        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Prepare input
            ref_s = voice[len(tokens)].clone()
            
            # Generate audio
            return forward(self._model, tokens, ref_s, speed)
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
        finally:
            # Clean up memory
            gc.collect()