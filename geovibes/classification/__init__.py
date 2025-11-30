"""
GeoVibes Classification Module

Provides XGBoost-based classification on satellite embedding vectors.

Note: ClassificationPipeline is not exported here to avoid circular import
warnings when running `python -m geovibes.classification.pipeline`.
Import it directly: `from geovibes.classification.pipeline import ClassificationPipeline`
"""

from geovibes.classification.data_loader import ClassificationDataLoader
from geovibes.classification.classifier import EmbeddingClassifier
from geovibes.classification.inference import BatchInference
from geovibes.classification.output import OutputGenerator
from geovibes.classification.sample_negatives import (
    NegativeSampler,
    LULCConfig,
    SamplingConfig,
)

__all__ = [
    "ClassificationDataLoader",
    "EmbeddingClassifier",
    "BatchInference",
    "OutputGenerator",
    "NegativeSampler",
    "LULCConfig",
    "SamplingConfig",
]
