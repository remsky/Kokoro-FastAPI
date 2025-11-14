"""
Audio quality detection and analysis for voice prompts.

Detects issues with voice prompts and provides recommendations.
"""

import numpy as np
import soundfile as sf
from loguru import logger
from typing import Dict, List, Tuple

from api.src.core.config import settings


class QualityDetectionService:
    """Detect and analyze audio quality for voice prompts."""

    def __init__(self):
        """Initialize quality detection service."""
        self.quality_threshold = settings.quality_threshold

    def analyze_audio(self, audio_path: str) -> Dict[str, any]:
        """Analyze audio file quality.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with quality metrics and recommendations
        """
        try:
            # Load audio
            audio, sample_rate = sf.read(audio_path)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Calculate metrics
            duration = len(audio) / sample_rate
            rms_level = np.sqrt(np.mean(audio**2))
            peak_level = np.max(np.abs(audio))
            dynamic_range = peak_level / (rms_level + 1e-10)

            # Detect silence
            silence_threshold = 0.01
            silent_samples = np.sum(np.abs(audio) < silence_threshold)
            silence_ratio = silent_samples / len(audio)

            # Detect clipping
            clipping_threshold = 0.99
            clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
            clipping_ratio = clipped_samples / len(audio)

            # Estimate SNR (simplified)
            # Assuming first/last 0.1s might be noise
            noise_samples = int(0.1 * sample_rate)
            if len(audio) > 2 * noise_samples:
                noise = np.concatenate([audio[:noise_samples], audio[-noise_samples:]])
                noise_power = np.mean(noise**2)
                signal_power = np.mean(audio**2)
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            else:
                snr = float('inf')

            # Calculate quality score (0-1)
            quality_score = 1.0

            # Penalize for issues
            if duration < 0.5 or duration > settings.zipvoice_max_prompt_duration:
                quality_score -= 0.2
            if silence_ratio > 0.3:
                quality_score -= 0.2
            if clipping_ratio > 0.01:
                quality_score -= 0.3
            if rms_level < 0.05:
                quality_score -= 0.2
            if snr < 20 and snr != float('inf'):
                quality_score -= 0.2

            quality_score = max(0.0, quality_score)

            # Generate recommendations
            recommendations = []
            warnings = []

            if duration < 1.0:
                recommendations.append("Audio too short - use 1-3 seconds for better results")
            if duration > settings.zipvoice_max_prompt_duration:
                warnings.append(f"Audio exceeds maximum duration ({settings.zipvoice_max_prompt_duration}s)")
            if silence_ratio > 0.3:
                recommendations.append("Too much silence - trim or use clearer audio")
            if clipping_ratio > 0.01:
                warnings.append("Audio clipping detected - reduce input volume")
            if rms_level < 0.05:
                recommendations.append("Audio too quiet - increase volume")
            if snr < 20 and snr != float('inf'):
                recommendations.append("High background noise detected - use cleaner recording")
            if sample_rate not in [16000, 22050, 24000, 44100, 48000]:
                recommendations.append(f"Non-standard sample rate ({sample_rate}Hz) - may affect quality")

            result = {
                'quality_score': quality_score,
                'duration': duration,
                'sample_rate': sample_rate,
                'metrics': {
                    'rms_level': float(rms_level),
                    'peak_level': float(peak_level),
                    'dynamic_range': float(dynamic_range),
                    'silence_ratio': float(silence_ratio),
                    'clipping_ratio': float(clipping_ratio),
                    'snr_db': float(snr) if snr != float('inf') else None
                },
                'passes_threshold': quality_score >= self.quality_threshold,
                'recommendations': recommendations,
                'warnings': warnings
            }

            logger.debug(f"Quality analysis: score={quality_score:.2f}, duration={duration:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return {
                'quality_score': 0.0,
                'error': str(e),
                'passes_threshold': False,
                'recommendations': ['Failed to analyze audio'],
                'warnings': ['Audio analysis error']
            }

    def validate_prompt_quality(self, audio_path: str) -> Tuple[bool, List[str]]:
        """Quick validation of prompt audio quality.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        analysis = self.analyze_audio(audio_path)

        if analysis.get('error'):
            return False, [analysis['error']]

        issues = []

        if not analysis['passes_threshold']:
            issues.append(f"Quality score {analysis['quality_score']:.2f} below threshold {self.quality_threshold}")

        issues.extend(analysis.get('warnings', []))

        is_valid = len(issues) == 0 and analysis['passes_threshold']

        return is_valid, issues

    def suggest_improvements(self, audio_path: str) -> List[str]:
        """Get list of suggestions to improve audio quality.

        Args:
            audio_path: Path to audio file

        Returns:
            List of improvement suggestions
        """
        analysis = self.analyze_audio(audio_path)

        if analysis.get('error'):
            return ["Fix audio file errors"]

        suggestions = []

        # Add all recommendations
        suggestions.extend(analysis.get('recommendations', []))

        # Add quality-specific suggestions
        if analysis['quality_score'] < 0.5:
            suggestions.insert(0, "Consider re-recording with better equipment")
        elif analysis['quality_score'] < 0.7:
            suggestions.insert(0, "Audio quality could be improved")

        return suggestions

    def compare_prompts(
        self,
        audio_path1: str,
        audio_path2: str
    ) -> Dict[str, any]:
        """Compare quality of two audio prompts.

        Args:
            audio_path1: First audio file
            audio_path2: Second audio file

        Returns:
            Comparison results
        """
        analysis1 = self.analyze_audio(audio_path1)
        analysis2 = self.analyze_audio(audio_path2)

        better = None
        if analysis1['quality_score'] > analysis2['quality_score']:
            better = 'first'
        elif analysis2['quality_score'] > analysis1['quality_score']:
            better = 'second'
        else:
            better = 'equal'

        return {
            'first': analysis1,
            'second': analysis2,
            'better': better,
            'score_difference': abs(analysis1['quality_score'] - analysis2['quality_score'])
        }


# Global instance
_quality_detection_service = None


def get_quality_detection_service() -> QualityDetectionService:
    """Get global quality detection service instance."""
    global _quality_detection_service

    if _quality_detection_service is None:
        _quality_detection_service = QualityDetectionService()

    return _quality_detection_service
