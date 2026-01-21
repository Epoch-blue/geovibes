"""Pipeline orchestrator for batch verification of detections."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import geopandas as gpd

from geovibes.agents.clustering import cluster_detections_from_file, cluster_detections
from geovibes.agents.schemas import (
    ClusterInfo,
    VerificationDecision,
    VerificationResult,
    VerifiedFacility,
)
from geovibes.agents.verification_agent import VerificationAgent


@dataclass
class PipelineConfig:
    """Configuration for the verification pipeline."""

    target_type: str = "palm_oil_mill"
    target_prompt: Optional[str] = None
    reference_image_path: Optional[str] = None
    cluster_eps_m: float = 500.0
    cluster_min_samples: int = 2
    min_probability: Optional[float] = None
    confidence_threshold: float = 0.7
    enable_places: bool = True
    max_workers: int = 4
    gemini_api_key: Optional[str] = None
    google_maps_api_key: Optional[str] = None


@dataclass
class PipelineResults:
    """Results from running the verification pipeline."""

    total_clusters: int
    verified_valid: int
    verified_invalid: int
    results: list[VerificationResult]
    facilities: list[VerifiedFacility]
    processing_time_sec: float


class VerificationPipeline:
    """Pipeline for batch verification of detection clusters."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the verification pipeline.

        Parameters
        ----------
        config : PipelineConfig, optional
            Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self._agent: Optional[VerificationAgent] = None

    @property
    def agent(self) -> VerificationAgent:
        """Lazily initialize the verification agent."""
        if self._agent is None:
            self._agent = VerificationAgent(
                target_type=self.config.target_type,
                target_prompt=self.config.target_prompt,
                reference_image_path=self.config.reference_image_path,
                gemini_api_key=self.config.gemini_api_key,
                google_maps_api_key=self.config.google_maps_api_key,
                confidence_threshold=self.config.confidence_threshold,
                enable_places=self.config.enable_places,
            )
        return self._agent

    def run_from_geojson(
        self,
        detections_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PipelineResults:
        """
        Run verification pipeline on detections from a GeoJSON file.

        Parameters
        ----------
        detections_path : str
            Path to detections GeoJSON
        output_path : str, optional
            Path to write verified facilities GeoJSON
        progress_callback : callable, optional
            Called with (completed, total) after each cluster

        Returns
        -------
        PipelineResults
            Pipeline execution results
        """
        clusters = cluster_detections_from_file(
            detections_path,
            eps_m=self.config.cluster_eps_m,
            min_samples=self.config.cluster_min_samples,
            min_probability=self.config.min_probability,
        )

        results = self._process_clusters(clusters, progress_callback)

        if output_path:
            self._export_results(results, output_path)

        return results

    def run_from_gdf(
        self,
        detections_gdf: gpd.GeoDataFrame,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PipelineResults:
        """
        Run verification pipeline on detections from a GeoDataFrame.

        Parameters
        ----------
        detections_gdf : gpd.GeoDataFrame
            GeoDataFrame with detections
        output_path : str, optional
            Path to write verified facilities GeoJSON
        progress_callback : callable, optional
            Called with (completed, total) after each cluster

        Returns
        -------
        PipelineResults
            Pipeline execution results
        """
        if (
            self.config.min_probability is not None
            and "probability" in detections_gdf.columns
        ):
            detections_gdf = detections_gdf[
                detections_gdf["probability"] >= self.config.min_probability
            ].copy()

        clusters = cluster_detections(
            detections_gdf,
            eps_m=self.config.cluster_eps_m,
            min_samples=self.config.cluster_min_samples,
        )

        results = self._process_clusters(clusters, progress_callback)

        if output_path:
            self._export_results(results, output_path)

        return results

    def _process_clusters(
        self,
        clusters: list[ClusterInfo],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PipelineResults:
        """Process clusters through the verification agent."""
        import time

        start_time = time.perf_counter()
        total = len(clusters)
        results: list[VerificationResult] = []
        completed = 0

        if self.config.max_workers > 1 and total > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self.agent.verify_cluster, cluster): cluster
                    for cluster in clusters
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        cluster = futures[future]
                        results.append(
                            VerificationResult(
                                cluster=cluster,
                                decision=VerificationDecision.INVALID,
                                rejection_reason=f"Processing error: {str(e)}",
                            )
                        )

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
        else:
            for cluster in clusters:
                try:
                    result = self.agent.verify_cluster(cluster)
                    results.append(result)
                except Exception as e:
                    results.append(
                        VerificationResult(
                            cluster=cluster,
                            decision=VerificationDecision.INVALID,
                            rejection_reason=f"Processing error: {str(e)}",
                        )
                    )

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        elapsed = time.perf_counter() - start_time

        valid_count = sum(
            1 for r in results if r.decision == VerificationDecision.VALID
        )
        invalid_count = sum(
            1 for r in results if r.decision == VerificationDecision.INVALID
        )
        facilities = [r.facility for r in results if r.facility is not None]

        return PipelineResults(
            total_clusters=total,
            verified_valid=valid_count,
            verified_invalid=invalid_count,
            results=results,
            facilities=facilities,
            processing_time_sec=elapsed,
        )

    def _export_results(self, results: PipelineResults, output_path: str) -> None:
        """Export results to GeoJSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        features = []
        for facility in results.facilities:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [facility.lon, facility.lat],
                },
                "properties": {
                    "company_name": facility.company_name,
                    "facility_name": facility.facility_name,
                    "facility_type": facility.facility_type,
                    "address": facility.address,
                    "confidence": facility.confidence,
                    "notes": facility.notes,
                    "source_urls": facility.source_urls,
                },
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_clusters_processed": results.total_clusters,
                "verified_valid": results.verified_valid,
                "verified_invalid": results.verified_invalid,
                "processing_time_sec": results.processing_time_sec,
            },
        }

        with open(path, "w") as f:
            json.dump(geojson, f, indent=2)

        rejected_path = path.with_stem(path.stem + "_rejected")
        rejected_features = []
        for result in results.results:
            if result.decision == VerificationDecision.INVALID:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            result.cluster.centroid_lon,
                            result.cluster.centroid_lat,
                        ],
                    },
                    "properties": {
                        "cluster_id": result.cluster.cluster_id,
                        "detection_count": result.cluster.detection_count,
                        "avg_probability": result.cluster.avg_probability,
                        "rejection_reason": result.rejection_reason,
                        "alternative_facility_type": result.alternative_facility_type,
                    },
                }
                rejected_features.append(feature)

        if rejected_features:
            rejected_geojson = {
                "type": "FeatureCollection",
                "features": rejected_features,
            }
            with open(rejected_path, "w") as f:
                json.dump(rejected_geojson, f, indent=2)


def run_verification_pipeline(
    detections_path: str,
    output_path: str,
    target_type: str = "palm_oil_mill",
    cluster_eps_m: float = 500.0,
    min_probability: Optional[float] = None,
    max_workers: int = 4,
    verbose: bool = True,
) -> PipelineResults:
    """
    Convenience function to run the verification pipeline.

    Parameters
    ----------
    detections_path : str
        Path to detections GeoJSON
    output_path : str
        Path to write verified facilities GeoJSON
    target_type : str
        Type of facility to verify
    cluster_eps_m : float
        DBSCAN clustering radius in meters
    min_probability : float, optional
        Minimum probability threshold
    max_workers : int
        Number of parallel workers
    verbose : bool
        Whether to print progress

    Returns
    -------
    PipelineResults
        Pipeline execution results
    """
    config = PipelineConfig(
        target_type=target_type,
        cluster_eps_m=cluster_eps_m,
        min_probability=min_probability,
        max_workers=max_workers,
    )

    pipeline = VerificationPipeline(config)

    def progress(completed: int, total: int) -> None:
        if verbose:
            print(f"\rProcessing clusters: {completed}/{total}", end="", flush=True)

    results = pipeline.run_from_geojson(
        detections_path,
        output_path,
        progress_callback=progress if verbose else None,
    )

    if verbose:
        print()
        print(f"\n{'=' * 60}")
        print("VERIFICATION RESULTS")
        print(f"{'=' * 60}")
        print(f"Total clusters:     {results.total_clusters}")
        print(f"Verified valid:     {results.verified_valid}")
        print(f"Verified invalid:   {results.verified_invalid}")
        print(f"Processing time:    {results.processing_time_sec:.1f}s")
        print(f"Output:             {output_path}")
        print(f"{'=' * 60}")

    return results
