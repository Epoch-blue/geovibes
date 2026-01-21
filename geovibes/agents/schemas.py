"""Pydantic schemas for the verification agent."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VerificationDecision(str, Enum):
    """Decision made by the verification agent."""

    VALID = "valid"
    INVALID = "invalid"
    NEEDS_CONTEXT = "needs_context"


class ClusterInfo(BaseModel):
    """Information about a cluster of detections."""

    cluster_id: int = Field(description="Unique identifier for the cluster")
    centroid_lat: float = Field(description="Latitude of cluster centroid")
    centroid_lon: float = Field(description="Longitude of cluster centroid")
    bounds_min_lat: float = Field(description="Minimum latitude of cluster bounds")
    bounds_max_lat: float = Field(description="Maximum latitude of cluster bounds")
    bounds_min_lon: float = Field(description="Minimum longitude of cluster bounds")
    bounds_max_lon: float = Field(description="Maximum longitude of cluster bounds")
    detection_count: int = Field(description="Number of detections in the cluster")
    avg_probability: float = Field(description="Average probability of detections")
    max_probability: float = Field(description="Maximum probability in the cluster")
    detection_ids: list[int] = Field(
        default_factory=list, description="IDs of detections in this cluster"
    )


class PlaceInfo(BaseModel):
    """Information about a nearby place from Google Places API."""

    place_id: str = Field(description="Google Places place ID")
    name: str = Field(description="Name of the place")
    types: list[str] = Field(default_factory=list, description="Place types")
    address: str = Field(default="", description="Formatted address")
    distance_m: float = Field(description="Distance from cluster centroid in meters")
    rating: Optional[float] = Field(default=None, description="Place rating")
    lat: float = Field(description="Latitude of the place")
    lon: float = Field(description="Longitude of the place")


class VerifiedFacility(BaseModel):
    """Information about a verified facility."""

    company_name: Optional[str] = Field(
        default=None, description="Name of the company operating the facility"
    )
    facility_name: Optional[str] = Field(
        default=None, description="Name of the facility"
    )
    facility_type: str = Field(description="Type of facility identified")
    address: Optional[str] = Field(default=None, description="Address of the facility")
    lat: float = Field(description="Latitude of the facility")
    lon: float = Field(description="Longitude of the facility")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the verification"
    )
    source_urls: list[str] = Field(
        default_factory=list, description="URLs used to verify the facility"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional notes about the facility"
    )


class VerificationResult(BaseModel):
    """Result of verifying a cluster."""

    cluster: ClusterInfo = Field(description="The cluster that was verified")
    decision: VerificationDecision = Field(description="Verification decision")
    facility: Optional[VerifiedFacility] = Field(
        default=None, description="Verified facility info if decision is VALID"
    )
    rejection_reason: Optional[str] = Field(
        default=None, description="Reason for rejection if decision is INVALID"
    )
    alternative_facility_type: Optional[str] = Field(
        default=None, description="What the detection actually is if not target"
    )
    places_found: list[PlaceInfo] = Field(
        default_factory=list, description="Nearby places found during verification"
    )
    context_gathered: bool = Field(
        default=False, description="Whether additional context was gathered"
    )


class InitialVerificationOutput(BaseModel):
    """Structured output from initial verification step."""

    decision: VerificationDecision = Field(
        description="VALID if this appears to be the target facility type, "
        "INVALID if clearly not, NEEDS_CONTEXT if uncertain"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the decision (0-1)"
    )
    reasoning: str = Field(description="Brief explanation of the decision")
    observed_features: list[str] = Field(
        default_factory=list,
        description="List of features observed in the satellite image",
    )
    facility_type_guess: Optional[str] = Field(
        default=None,
        description="Best guess at what type of facility this is",
    )


class EnrichmentOutput(BaseModel):
    """Structured output from facility enrichment step."""

    company_name: Optional[str] = Field(
        default=None, description="Company name if identified"
    )
    facility_name: Optional[str] = Field(
        default=None, description="Facility name if identified"
    )
    address: Optional[str] = Field(default=None, description="Address if found")
    facility_type: str = Field(description="Confirmed facility type")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the identification"
    )
    evidence: list[str] = Field(
        default_factory=list, description="Evidence supporting the identification"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")
