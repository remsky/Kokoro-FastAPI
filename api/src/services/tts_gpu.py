import os
import time

import numpy as np
import torch
from loguru import logger
from models import build_model

from .tts_base import TTSBaseModel
from ..core.config import settings
from .text_processing import tokenize, phonemize


# @torch.no_grad()
# def forward(model, tokens, ref_s, speed):
#     """Forward pass through the model"""
#     device = ref_s.device
#     tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
#     input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
#     text_mask = length_to_mask(input_lengths).to(device)
#     bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
#     d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
#     s = ref_s[:, 128:]
#     d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
#     x, _ = model.predictor.lstm(d)
#     duration = model.predictor.duration_proj(x)
#     duration = torch.sigmoid(duration).sum(axis=-1) / speed
#     pred_dur = torch.round(duration).clamp(min=1).long()
#     pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
#     c_frame = 0
#     for i in range(pred_aln_trg.size(0)):
#         pred_aln_trg[i, c_frame : c_frame + pred_dur[0, i].item()] = 1
#         c_frame += pred_dur[0, i].item()
#     en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
#     F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
#     t_en = model.text_encoder(tokens, input_lengths, text_mask)
#     asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
#     return model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy()
@torch.no_grad()
def forward(model, tokens, ref_s, speed):
    """Forward pass through the model with moderate memory management"""
    device = ref_s.device
    
    try:
        # Initial tensor setup with proper device placement
        tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        # Split and clone reference signals with explicit device placement
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
        # Only cleanup large intermediates
        del duration, x

        # Alignment matrix construction
        pred_aln_trg = torch.zeros(input_lengths.item(), pred_dur.sum().item(), device=device)
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + pred_dur[0, i].item()] = 1
            c_frame += pred_dur[0, i].item()
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        # Matrix multiplications with selective cleanup
        en = d.transpose(-1, -2) @ pred_aln_trg
        del d  # Free large intermediate tensor
        
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s_content)
        del en  # Free large intermediate tensor

        # Final text encoding and decoding
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        del t_en  # Free large intermediate tensor

        # Final decoding and transfer to CPU
        output = model.decoder(asr, F0_pred, N_pred, s_ref)
        result = output.squeeze().cpu().numpy()
        
        return result
        
    finally:
        # Let PyTorch handle most cleanup automatically
        # Only explicitly free the largest tensors
        del pred_aln_trg, asr


# def length_to_mask(lengths):
#     """Create attention mask from lengths"""
#     mask = (
#         torch.arange(lengths.max())
#         .unsqueeze(0)
#         .expand(lengths.shape[0], -1)
#         .type_as(lengths)
#     )
#     mask = torch.gt(mask + 1, lengths.unsqueeze(1))
#     return mask


def length_to_mask(lengths):
    """Create attention mask from lengths - possibly optimized version"""
    max_len = lengths.max()
    # Create mask directly on the same device as lengths
    mask = torch.arange(max_len, device=lengths.device)[None, :].expand(
        lengths.shape[0], -1
    )
    # Avoid type_as by using the correct dtype from the start
    if lengths.dtype != mask.dtype:
        mask = mask.to(dtype=lengths.dtype)
    # Fuse operations  using broadcasting
    return mask + 1 > lengths[:, None]


class TTSGPUModel(TTSBaseModel):
    _instance = None
    _device = "cuda"

    @classmethod
    def get_instance(cls):
        """Get the model instance"""
        if cls._instance is None:
            raise RuntimeError("GPU model not initialized. Call initialize() first.")
        return cls._instance

    @classmethod
    def initialize(cls, model_dir: str, model_path: str):
        """Initialize PyTorch model for GPU inference"""
        if cls._instance is None and torch.cuda.is_available():
            try:
                logger.info("Initializing GPU model")
                model_path = os.path.join(model_dir, settings.pytorch_model_path)
                model = build_model(model_path, cls._device)
                cls._instance = model
                return model
            except Exception as e:
                logger.error(f"Failed to initialize GPU model: {e}")
                return None
        return cls._instance

    @classmethod
    def process_text(cls, text: str, language: str) -> tuple[str, list[int]]:
        """Process text into phonemes and tokens

        Args:
            text: Input text
            language: Language code

        Returns:
            tuple[str, list[int]]: Phonemes and token IDs
        """
        phonemes = phonemize(text, language)
        tokens = tokenize(phonemes)
        return phonemes, tokens

    @classmethod
    def generate_from_text(
        cls, text: str, voicepack: torch.Tensor, language: str, speed: float
    ) -> tuple[np.ndarray, str]:
        """Generate audio from text

        Args:
            text: Input text
            voicepack: Voice tensor
            language: Language code
            speed: Speed factor

        Returns:
            tuple[np.ndarray, str]: Generated audio samples and phonemes
        """
        if cls._instance is None:
            raise RuntimeError("GPU model not initialized")

        # Process text
        phonemes, tokens = cls.process_text(text, language)

        # Generate audio
        audio = cls.generate_from_tokens(tokens, voicepack, speed)

        return audio, phonemes

    @classmethod
    def generate_from_tokens(
        cls, tokens: list[int], voicepack: torch.Tensor, speed: float
    ) -> np.ndarray:
        """Generate audio from tokens with moderate memory management

        Args:
            tokens: Token IDs
            voicepack: Voice tensor
            speed: Speed factor

        Returns:
            np.ndarray: Generated audio samples
        """
        if cls._instance is None:
            raise RuntimeError("GPU model not initialized")

        try:
            device = cls._device
            
            # Check memory pressure
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # Convert to GB
                if memory_allocated > 2.0:  # 2GB limit
                    logger.info(
                        f"Memory usage above 2GB threshold:{memory_allocated:.2f}GB "
                        f"Clearing cache"
                    )
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            
            # Get reference style with proper device placement
            ref_s = voicepack[len(tokens)].clone().to(device)
            
            # Generate audio
            audio = forward(cls._instance, tokens, ref_s, speed)
            
            return audio
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # On OOM, do a full cleanup and retry
                if torch.cuda.is_available():
                    logger.warning("Out of memory detected, performing full cleanup")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
                    # Log memory stats after cleanup
                    memory_allocated = torch.cuda.memory_allocated(device)
                    memory_reserved = torch.cuda.memory_reserved(device)
                    logger.info(
                        f"Memory after OOM cleanup: "
                        f"Allocated: {memory_allocated / 1e9:.2f}GB, "
                        f"Reserved: {memory_reserved / 1e9:.2f}GB"
                    )
                    
                    # Retry generation
                    ref_s = voicepack[len(tokens)].clone().to(device)
                    audio = forward(cls._instance, tokens, ref_s, speed)
                    return audio
            raise
            
        finally:
            # Only synchronize at the top level, no empty_cache
            if torch.cuda.is_available():
                torch.cuda.synchronize()
