"""Google Places API wrapper for verification agent."""

import math
import os
from typing import Optional

from geovibes.agents.schemas import PlaceInfo

try:
    import googlemaps

    GOOGLEMAPS_AVAILABLE = True
except ImportError:
    googlemaps = None
    GOOGLEMAPS_AVAILABLE = False


class PlacesClient:
    """Client for Google Places API operations."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Places client.

        Parameters
        ----------
        api_key : str, optional
            Google Maps API key. Falls back to GOOGLE_MAPS_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")

        if GOOGLEMAPS_AVAILABLE and self.api_key:
            self.client = googlemaps.Client(key=self.api_key)
        else:
            self.client = None

    @property
    def is_available(self) -> bool:
        """Check if the client is available and configured."""
        return self.client is not None

    def search_nearby(
        self,
        lat: float,
        lon: float,
        radius_m: float = 1000.0,
        limit: int = 10,
    ) -> list[PlaceInfo]:
        """
        Search for nearby places.

        Parameters
        ----------
        lat : float
            Latitude of search center
        lon : float
            Longitude of search center
        radius_m : float
            Search radius in meters
        limit : int
            Maximum number of results

        Returns
        -------
        list[PlaceInfo]
            List of nearby places sorted by distance
        """
        if not self.client:
            return []

        try:
            result = self.client.places_nearby(
                location=(lat, lon),
                radius=int(radius_m),
            )

            places = []
            for place in result.get("results", []):
                place_info = self._parse_place(place, lat, lon)
                if place_info:
                    places.append(place_info)

            places.sort(key=lambda p: p.distance_m)
            return places[:limit]

        except Exception:
            return []

    def get_place_details(self, place_id: str) -> Optional[dict]:
        """
        Get detailed information for a place.

        Parameters
        ----------
        place_id : str
            Google Places place ID

        Returns
        -------
        dict or None
            Place details or None if unavailable
        """
        if not self.client:
            return None

        try:
            result = self.client.place(
                place_id=place_id,
                fields=[
                    "name",
                    "formatted_address",
                    "formatted_phone_number",
                    "website",
                    "url",
                    "types",
                    "business_status",
                    "opening_hours",
                    "photos",
                    "reviews",
                ],
            )
            return result.get("result")
        except Exception:
            return None

    def get_place_photo_url(
        self,
        photo_reference: str,
        max_width: int = 800,
    ) -> Optional[str]:
        """
        Get URL for a place photo.

        Parameters
        ----------
        photo_reference : str
            Photo reference from place details
        max_width : int
            Maximum width in pixels

        Returns
        -------
        str or None
            Photo URL or None if unavailable
        """
        if not self.api_key:
            return None

        return (
            f"https://maps.googleapis.com/maps/api/place/photo"
            f"?maxwidth={max_width}"
            f"&photo_reference={photo_reference}"
            f"&key={self.api_key}"
        )

    def _parse_place(
        self,
        place_data: dict,
        ref_lat: float,
        ref_lon: float,
    ) -> Optional[PlaceInfo]:
        """Parse raw place data into PlaceInfo."""
        try:
            geometry = place_data.get("geometry", {})
            location = geometry.get("location", {})
            place_lat = location.get("lat")
            place_lon = location.get("lng")

            if place_lat is None or place_lon is None:
                return None

            distance = self._haversine_distance(ref_lat, ref_lon, place_lat, place_lon)

            return PlaceInfo(
                place_id=place_data.get("place_id", ""),
                name=place_data.get("name", "Unknown"),
                types=place_data.get("types", []),
                address=place_data.get("vicinity", ""),
                distance_m=distance,
                rating=place_data.get("rating"),
                lat=place_lat,
                lon=place_lon,
            )
        except Exception:
            return None

    @staticmethod
    def _haversine_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two points in meters."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lat_diff = math.radians(lat2 - lat1)
        lon_diff = math.radians(lon2 - lon1)

        a = (
            math.sin(lat_diff / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(lon_diff / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        return 6371000 * c


def format_places_for_prompt(places: list[PlaceInfo]) -> str:
    """
    Format places list for inclusion in LLM prompt.

    Parameters
    ----------
    places : list[PlaceInfo]
        List of places to format

    Returns
    -------
    str
        Formatted string for prompt
    """
    if not places:
        return "No nearby places found within search radius."

    lines = ["Nearby places found:"]
    for i, place in enumerate(places, 1):
        types_str = ", ".join(place.types[:5]) if place.types else "unknown"
        distance_str = (
            f"{place.distance_m:.0f}m"
            if place.distance_m < 1000
            else f"{place.distance_m / 1000:.1f}km"
        )
        lines.append(f"{i}. {place.name}")
        lines.append(f"   Distance: {distance_str}")
        lines.append(f"   Types: {types_str}")
        if place.address:
            lines.append(f"   Address: {place.address}")
        lines.append("")

    return "\n".join(lines)
