"""Verification agent for validating classification pipeline detections."""

from geovibes.agents.schemas import (
    ClusterInfo,
    PlaceInfo,
    VerificationDecision,
    VerificationResult,
    VerifiedFacility,
)
from geovibes.agents.clustering import (
    cluster_detections,
    cluster_detections_from_file,
    create_single_cluster,
)
from geovibes.agents.verification_agent import VerificationAgent
from geovibes.agents.pipeline import (
    PipelineConfig,
    PipelineResults,
    VerificationPipeline,
    run_verification_pipeline,
)

__all__ = [
    # Schemas
    "ClusterInfo",
    "PlaceInfo",
    "VerificationDecision",
    "VerificationResult",
    "VerifiedFacility",
    # Clustering
    "cluster_detections",
    "cluster_detections_from_file",
    "create_single_cluster",
    # Agent
    "VerificationAgent",
    # Pipeline
    "PipelineConfig",
    "PipelineResults",
    "VerificationPipeline",
    "run_verification_pipeline",
]
