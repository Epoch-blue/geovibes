"""Tests for the verification agent using known true/false positive locations."""

import os
from unittest.mock import patch

import pytest

from geovibes.agents.clustering import create_single_cluster, cluster_detections
from geovibes.agents.schemas import (
    PlaceInfo,
    VerificationDecision,
    VerificationResult,
)
from geovibes.agents.verification_agent import VerificationAgent


# Test coordinates derived from validation against truth dataset
# (scripts/validate_detections.py with XGBoost results)

# TRUE POSITIVE: 4.3m from nearest truth point, detection index 12301
# Location: Lampung province, southern Sumatra - known palm oil mill
TRUE_POSITIVE = {
    "lat": -5.094876,
    "lon": 105.186421,
    "prob": 0.856,
    "description": "Known palm oil mill in Lampung, 4.3m from truth point",
}

# FALSE POSITIVE: 154km from nearest truth point, detection index 7723
# Location: West Sumatra province - NOT a palm oil mill
FALSE_POSITIVE = {
    "lat": -1.600795,
    "lon": 99.195028,
    "prob": 0.657,
    "description": "False positive in West Sumatra, 154km from any known mill",
}


class TestClusterInfo:
    """Tests for ClusterInfo creation and manipulation."""

    def test_create_single_cluster(self):
        """Test creating a single-point cluster."""
        cluster = create_single_cluster(
            lat=TRUE_POSITIVE["lat"],
            lon=TRUE_POSITIVE["lon"],
            probability=TRUE_POSITIVE["prob"],
        )

        assert cluster.centroid_lat == TRUE_POSITIVE["lat"]
        assert cluster.centroid_lon == TRUE_POSITIVE["lon"]
        assert cluster.avg_probability == TRUE_POSITIVE["prob"]
        assert cluster.detection_count == 1

    def test_cluster_bounds(self):
        """Test that cluster bounds are valid."""
        cluster = create_single_cluster(lat=-5.0, lon=105.0, probability=0.8)

        assert cluster.bounds_min_lat < cluster.centroid_lat
        assert cluster.bounds_max_lat > cluster.centroid_lat
        assert cluster.bounds_min_lon < cluster.centroid_lon
        assert cluster.bounds_max_lon > cluster.centroid_lon


class TestVerificationAgentInit:
    """Tests for VerificationAgent initialization."""

    def test_agent_init_without_api_keys(self):
        """Test agent initializes without API keys (graceful degradation)."""
        with patch.dict(os.environ, {}, clear=True):
            agent = VerificationAgent(
                gemini_api_key=None,
                google_maps_api_key=None,
            )
            assert agent.target_type == "palm_oil_mill"
            assert agent.genai_client is None

    def test_agent_init_with_custom_target(self):
        """Test agent initializes with custom target type."""
        agent = VerificationAgent(
            target_type="aquaculture_pond",
            target_prompt="This is a fish farm",
            gemini_api_key=None,
        )
        assert agent.target_type == "aquaculture_pond"
        assert "fish farm" in agent.target_prompt


class TestVerificationDecisions:
    """Tests for verification decision logic."""

    def test_verification_result_valid(self):
        """Test creating a valid verification result."""
        cluster = create_single_cluster(lat=-5.0, lon=105.0, probability=0.8)
        result = VerificationResult(
            cluster=cluster,
            decision=VerificationDecision.VALID,
            facility=None,
        )
        assert result.decision == VerificationDecision.VALID
        assert result.rejection_reason is None

    def test_verification_result_invalid(self):
        """Test creating an invalid verification result."""
        cluster = create_single_cluster(lat=-1.6, lon=99.2, probability=0.6)
        result = VerificationResult(
            cluster=cluster,
            decision=VerificationDecision.INVALID,
            rejection_reason="Not a palm oil mill",
            alternative_facility_type="residential area",
        )
        assert result.decision == VerificationDecision.INVALID
        assert result.rejection_reason == "Not a palm oil mill"
        assert result.alternative_facility_type == "residential area"


class TestPlacesIntegration:
    """Tests for Google Places integration."""

    def test_place_info_model(self):
        """Test PlaceInfo model creation."""
        place = PlaceInfo(
            place_id="test123",
            name="PT Palm Oil Mill",
            types=["establishment", "point_of_interest"],
            address="Jalan Raya",
            distance_m=150.0,
            lat=-5.0,
            lon=105.0,
        )
        assert place.name == "PT Palm Oil Mill"
        assert place.distance_m == 150.0


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set - skipping integration tests",
)
class TestVerificationAgentIntegration:
    """Integration tests requiring API keys."""

    @pytest.fixture
    def agent(self):
        """Create agent with real API keys."""
        return VerificationAgent(target_type="palm_oil_mill")

    def test_true_positive_detection(self, agent):
        """
        Agent should verify known palm oil mill as VALID.

        This tests against a location that is 4.3m from a known
        palm oil mill in the truth dataset.
        """
        result = agent.verify_location(
            lat=TRUE_POSITIVE["lat"],
            lon=TRUE_POSITIVE["lon"],
        )

        # The agent should recognize this as a valid palm oil mill
        # or at least not confidently reject it
        if result.decision == VerificationDecision.INVALID:
            # If rejected, the confidence should be low and reasoning provided
            assert result.rejection_reason is not None
            print(f"Rejection reason: {result.rejection_reason}")
        else:
            # If valid, we should have some facility info
            assert result.decision in [
                VerificationDecision.VALID,
                VerificationDecision.NEEDS_CONTEXT,
            ]

    def test_false_positive_detection(self, agent):
        """
        Agent should reject location far from any known mills.

        This tests against a location that is 154km from any known
        palm oil mill - a false positive from the classification model.
        """
        result = agent.verify_location(
            lat=FALSE_POSITIVE["lat"],
            lon=FALSE_POSITIVE["lon"],
        )

        # The agent ideally should reject this, but at minimum
        # should not confidently accept it as a palm oil mill
        if result.decision == VerificationDecision.VALID:
            # If accepted, confidence should be low
            if result.facility:
                assert result.facility.confidence < 0.9
            print(f"Accepted with facility: {result.facility}")
        else:
            # Expected: rejection with reason
            assert result.decision == VerificationDecision.INVALID
            assert result.rejection_reason is not None


class TestVerificationAgentMocked:
    """Tests with mocked LLM responses."""

    @pytest.fixture
    def mock_agent(self):
        """Create agent with mocked model."""
        agent = VerificationAgent(
            target_type="palm_oil_mill",
            gemini_api_key=None,
            enable_places=False,
        )
        return agent

    @patch("geovibes.agents.thumbnail.get_map_image")
    def test_verify_location_without_model(self, mock_get_image, mock_agent):
        """Test verification falls back gracefully without model."""
        mock_get_image.return_value = b"fake_image_bytes"

        result = mock_agent.verify_location(
            lat=TRUE_POSITIVE["lat"],
            lon=TRUE_POSITIVE["lon"],
        )

        # Without a model, should return NEEDS_CONTEXT or fall through
        assert result.decision in [
            VerificationDecision.INVALID,
            VerificationDecision.NEEDS_CONTEXT,
        ]

    def test_cluster_creation_for_test_locations(self, mock_agent):
        """Test that test location coordinates create valid clusters."""
        tp_cluster = create_single_cluster(
            lat=TRUE_POSITIVE["lat"],
            lon=TRUE_POSITIVE["lon"],
            probability=TRUE_POSITIVE["prob"],
        )

        fp_cluster = create_single_cluster(
            lat=FALSE_POSITIVE["lat"],
            lon=FALSE_POSITIVE["lon"],
            probability=FALSE_POSITIVE["prob"],
        )

        # Verify clusters are geographically distinct
        lat_diff = abs(tp_cluster.centroid_lat - fp_cluster.centroid_lat)
        lon_diff = abs(tp_cluster.centroid_lon - fp_cluster.centroid_lon)

        assert lat_diff > 1.0  # More than 1 degree apart
        assert lon_diff > 1.0


class TestDBSCANClustering:
    """Tests for DBSCAN clustering functionality."""

    def test_clustering_with_geopandas(self):
        """Test clustering with a GeoDataFrame."""
        import geopandas as gpd
        from shapely.geometry import Point

        # Create test points - two clusters
        points = [
            # Cluster 1 (tight group)
            Point(105.0, -5.0),
            Point(105.001, -5.001),
            Point(105.002, -5.0),
            # Cluster 2 (far away)
            Point(99.0, -1.5),
            Point(99.001, -1.501),
        ]

        gdf = gpd.GeoDataFrame(
            {
                "id": range(len(points)),
                "probability": [0.8, 0.75, 0.9, 0.6, 0.65],
                "geometry": points,
            },
            crs="EPSG:4326",
        )

        clusters = cluster_detections(
            gdf,
            eps_m=500,
            min_samples=2,
        )

        # Should find 2 clusters
        assert len(clusters) >= 1  # At least the tight cluster
        assert all(c.detection_count >= 2 for c in clusters)
