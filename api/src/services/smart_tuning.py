"""
Smart parameter auto-tuning for ZipVoice generation.

Automatically optimizes generation parameters based on input characteristics.
"""

import re
from typing import Dict, Tuple

from loguru import logger

from api.src.core.config import settings


class SmartTuningService:
    """Automatically tune generation parameters for optimal quality/speed balance."""

    def __init__(self):
        """Initialize smart tuning service."""
        self.default_num_steps = settings.zipvoice_num_steps

    def analyze_text(self, text: str) -> Dict[str, any]:
        """Analyze input text characteristics.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with text characteristics
        """
        # Basic metrics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))

        # Complexity indicators
        has_numbers = bool(re.search(r'\d', text))
        has_special_chars = bool(re.search(r'[^\w\s.!?,]', text))
        avg_word_length = char_count / max(word_count, 1)

        # Estimate complexity score (0-1)
        complexity = 0.0
        if has_numbers:
            complexity += 0.2
        if has_special_chars:
            complexity += 0.2
        if avg_word_length > 6:
            complexity += 0.3
        if word_count > 100:
            complexity += 0.3

        complexity = min(complexity, 1.0)

        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'complexity': complexity,
            'has_numbers': has_numbers,
            'has_special_chars': has_special_chars
        }

    def recommend_parameters(
        self,
        text: str,
        priority: str = "balanced"  # "speed", "balanced", "quality"
    ) -> Dict[str, any]:
        """Recommend optimal generation parameters.

        Args:
            text: Input text
            priority: Optimization priority

        Returns:
            Dictionary with recommended parameters
        """
        analysis = self.analyze_text(text)

        # Base recommendations
        recommendations = {
            'model': settings.zipvoice_model,
            'num_steps': self.default_num_steps,
            'remove_long_silence': True,
            'speed': 1.0
        }

        # Adjust based on text length
        if analysis['word_count'] < 20:
            # Short text - can afford higher quality
            if priority != "speed":
                recommendations['num_steps'] = min(12, self.default_num_steps + 4)
        elif analysis['word_count'] > 100:
            # Long text - prioritize speed
            recommendations['num_steps'] = max(4, self.default_num_steps - 2)

        # Adjust based on priority
        if priority == "speed":
            recommendations['model'] = "zipvoice_distill"
            recommendations['num_steps'] = 4
        elif priority == "quality":
            recommendations['model'] = "zipvoice"
            recommendations['num_steps'] = 16
        else:  # balanced
            if analysis['complexity'] > 0.6:
                # Complex text - use more steps
                recommendations['num_steps'] = min(12, self.default_num_steps + 2)

        logger.debug(f"Smart tuning: {analysis} -> {recommendations}")

        return recommendations

    def estimate_generation_time(
        self,
        text: str,
        num_steps: int = None,
        model: str = None
    ) -> float:
        """Estimate generation time in seconds.

        Args:
            text: Input text
            num_steps: Number of inference steps
            model: Model variant

        Returns:
            Estimated time in seconds
        """
        analysis = self.analyze_text(text)
        num_steps = num_steps or self.default_num_steps
        model = model or settings.zipvoice_model

        # Base time per word (seconds)
        base_time_per_word = 0.1

        # Model speed multiplier
        model_multiplier = {
            'zipvoice_distill': 0.5,
            'zipvoice': 1.0,
            'zipvoice_dialog': 1.2,
            'zipvoice_dialog_stereo': 1.5
        }.get(model, 1.0)

        # Steps multiplier
        steps_multiplier = num_steps / 8.0

        # Calculate estimate
        estimated_time = (
            analysis['word_count'] *
            base_time_per_word *
            model_multiplier *
            steps_multiplier
        )

        # Add overhead
        overhead = 2.0  # seconds
        estimated_time += overhead

        return estimated_time

    def optimize_for_latency(self, text: str) -> Dict[str, any]:
        """Optimize parameters for minimum latency.

        Args:
            text: Input text

        Returns:
            Optimized parameters
        """
        return self.recommend_parameters(text, priority="speed")

    def optimize_for_quality(self, text: str) -> Dict[str, any]:
        """Optimize parameters for maximum quality.

        Args:
            text: Input text

        Returns:
            Optimized parameters
        """
        return self.recommend_parameters(text, priority="quality")


# Global instance
_smart_tuning_service = None


def get_smart_tuning_service() -> SmartTuningService:
    """Get global smart tuning service instance."""
    global _smart_tuning_service

    if _smart_tuning_service is None:
        _smart_tuning_service = SmartTuningService()

    return _smart_tuning_service
