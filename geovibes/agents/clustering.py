"""Spatial clustering of detections using DBSCAN."""

from typing import Optional

import geopandas as gpd
import numpy as np
import pyproj
from sklearn.cluster import DBSCAN

from geovibes.agents.schemas import ClusterInfo


def get_utm_crs(lat: float, lon: float) -> pyproj.CRS:
    """Compute UTM CRS from a point coordinate."""
    zone = int(((lon + 180) / 6) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return pyproj.CRS.from_epsg(epsg)


def cluster_detections(
    detections_gdf: gpd.GeoDataFrame,
    eps_m: float = 500.0,
    min_samples: int = 2,
    probability_column: str = "probability",
    id_column: str = "id",
) -> list[ClusterInfo]:
    """
    Cluster detections using DBSCAN.

    Parameters
    ----------
    detections_gdf : gpd.GeoDataFrame
        GeoDataFrame with detection geometries (points or polygons)
    eps_m : float
        Maximum distance between samples in meters for DBSCAN
    min_samples : int
        Minimum samples in a neighborhood to form a cluster
    probability_column : str
        Name of the probability column
    id_column : str
        Name of the ID column

    Returns
    -------
    list[ClusterInfo]
        List of cluster information objects
    """
    if len(detections_gdf) == 0:
        return []

    working_gdf = detections_gdf.copy()

    if working_gdf.geometry.iloc[0].geom_type != "Point":
        working_gdf["centroid"] = working_gdf.geometry.centroid
        centroids = working_gdf.set_geometry("centroid")
    else:
        centroids = working_gdf

    bounds = centroids.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    utm_crs = get_utm_crs(center_lat, center_lon)

    centroids_utm = centroids.to_crs(utm_crs)

    coords = np.array(
        [[geom.x, geom.y] for geom in centroids_utm.geometry if geom is not None]
    )

    if len(coords) == 0:
        return []

    clustering = DBSCAN(eps=eps_m, min_samples=min_samples, metric="euclidean")
    labels = clustering.fit_predict(coords)

    working_gdf["cluster_id"] = labels

    clusters = []
    unique_labels = set(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue

        mask = working_gdf["cluster_id"] == cluster_id
        cluster_gdf = working_gdf[mask]

        if "centroid" in cluster_gdf.columns:
            cluster_points = cluster_gdf.set_geometry("centroid")
        else:
            cluster_points = cluster_gdf

        bounds = cluster_points.total_bounds

        centroid_lon = (bounds[0] + bounds[2]) / 2
        centroid_lat = (bounds[1] + bounds[3]) / 2

        probs = cluster_gdf[probability_column].values
        detection_ids = cluster_gdf[id_column].tolist()

        cluster_info = ClusterInfo(
            cluster_id=int(cluster_id),
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            bounds_min_lat=bounds[1],
            bounds_max_lat=bounds[3],
            bounds_min_lon=bounds[0],
            bounds_max_lon=bounds[2],
            detection_count=len(cluster_gdf),
            avg_probability=float(np.mean(probs)),
            max_probability=float(np.max(probs)),
            detection_ids=[int(x) for x in detection_ids],
        )
        clusters.append(cluster_info)

    clusters.sort(key=lambda c: c.max_probability, reverse=True)

    return clusters


def cluster_detections_from_file(
    geojson_path: str,
    eps_m: float = 500.0,
    min_samples: int = 2,
    min_probability: Optional[float] = None,
) -> list[ClusterInfo]:
    """
    Load detections from GeoJSON and cluster them.

    Parameters
    ----------
    geojson_path : str
        Path to detection GeoJSON file
    eps_m : float
        Maximum distance between samples in meters for DBSCAN
    min_samples : int
        Minimum samples in a neighborhood to form a cluster
    min_probability : float, optional
        Minimum probability threshold to include detections

    Returns
    -------
    list[ClusterInfo]
        List of cluster information objects
    """
    gdf = gpd.read_file(geojson_path)

    if min_probability is not None and "probability" in gdf.columns:
        gdf = gdf[gdf["probability"] >= min_probability].copy()

    id_column = "id" if "id" in gdf.columns else "tile_id"
    if id_column not in gdf.columns:
        gdf["id"] = range(len(gdf))
        id_column = "id"

    return cluster_detections(
        gdf,
        eps_m=eps_m,
        min_samples=min_samples,
        id_column=id_column,
    )


def create_single_cluster(
    lat: float,
    lon: float,
    probability: float = 0.5,
    cluster_id: int = 0,
) -> ClusterInfo:
    """
    Create a single-point cluster for testing or single-location verification.

    Parameters
    ----------
    lat : float
        Latitude of the point
    lon : float
        Longitude of the point
    probability : float
        Detection probability
    cluster_id : int
        Cluster ID to assign

    Returns
    -------
    ClusterInfo
        Cluster containing a single point
    """
    buffer = 0.001

    return ClusterInfo(
        cluster_id=cluster_id,
        centroid_lat=lat,
        centroid_lon=lon,
        bounds_min_lat=lat - buffer,
        bounds_max_lat=lat + buffer,
        bounds_min_lon=lon - buffer,
        bounds_max_lon=lon + buffer,
        detection_count=1,
        avg_probability=probability,
        max_probability=probability,
        detection_ids=[0],
    )
