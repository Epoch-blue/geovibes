"""
Enhanced location analyzer that finds nearby Google Places and uses AI for analysis.
"""

import os
import math
from typing import Dict, Optional, List
from datetime import datetime

try:
    import googlemaps

    GOOGLEMAPS_AVAILABLE = True
except ImportError:
    googlemaps = None
    GOOGLEMAPS_AVAILABLE = False

try:
    from google import genai

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False


class LocationAnalyzer:
    """
    Enhanced location analyzer that finds nearby Google Places
    and provides AI-powered analysis.
    """

    def __init__(
        self,
        google_maps_api_key: str = None,
        gemini_api_key: str = None,
        model_name: str = "gemini-3-flash-preview",
    ):
        """
        Initialize the location analyzer with API keys.

        Args:
            google_maps_api_key: Google Maps API key for Places API
            gemini_api_key: Google AI API key for Gemini analysis
            model_name: Gemini model to use (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
        """
        self.google_maps_api_key = google_maps_api_key or os.getenv(
            "GOOGLE_MAPS_API_KEY"
        )
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

        # Initialize Google Maps client
        if GOOGLEMAPS_AVAILABLE and self.google_maps_api_key:
            self.gmaps = googlemaps.Client(key=self.google_maps_api_key)
        else:
            self.gmaps = None
            if not GOOGLEMAPS_AVAILABLE:
                pass  # Silently skip if package not installed
            elif not self.google_maps_api_key:
                print("Warning: Google Maps API key not found")

        # Initialize Gemini AI
        self.genai_client = None
        if GENAI_AVAILABLE:
            if self.gemini_api_key:
                self.genai_client = genai.Client(api_key=self.gemini_api_key)
            else:
                try:
                    self.genai_client = genai.Client()
                except ValueError:
                    self.genai_client = None
                    if not self.gemini_api_key:
                        print("Warning: Gemini API key not found")

    def find_nearby_places(self, lat: float, lon: float, limit: int = 5) -> List[Dict]:
        """
        Find the closest Google Places within 1km radius, sorted by distance.

        Args:
            lat: Latitude
            lon: Longitude
            limit: Number of closest places to return (default: 5)

        Returns:
            List of closest places sorted by distance, with their information
        """
        if not self.gmaps:
            return []

        try:
            # Use Google Places API to find nearby places within 1km
            places_result = self.gmaps.places_nearby(
                location=(lat, lon),
                radius=1000,  # 1km radius
                # No type restriction = get all types of places
            )

            # Calculate distance for each place and sort by distance
            nearby_places = []
            for place in places_result.get("results", []):
                place_geom = place.get("geometry", {})
                place_location = place_geom.get("location", {})
                place_lat = place_location.get("lat", lat)
                place_lon = place_location.get("lng", lon)

                # Calculate approximate distance in meters using Haversine-like formula
                # Simple distance calculation (good enough for sorting)
                lat_diff = math.radians(place_lat - lat)
                lon_diff = math.radians(place_lon - lon)
                a = (
                    math.sin(lat_diff / 2) ** 2
                    + math.cos(math.radians(lat))
                    * math.cos(math.radians(place_lat))
                    * math.sin(lon_diff / 2) ** 2
                )
                c = 2 * math.asin(math.sqrt(a))
                distance_m = 6371000 * c  # Earth radius in meters

                place_info = {
                    "name": place.get("name", "Unknown"),
                    "place_id": place.get("place_id", ""),
                    "types": place.get("types", []),
                    "vicinity": place.get("vicinity", ""),
                    "rating": place.get("rating", None),
                    "price_level": place.get("price_level", None),
                    "geometry": place.get("geometry", {}),
                    "formatted_address": place.get("vicinity", "No address"),
                    "distance_m": distance_m,  # Store distance for sorting
                }
                nearby_places.append(place_info)

            # Sort by distance and return top N closest
            nearby_places.sort(key=lambda x: x.get("distance_m", float("inf")))
            return nearby_places[:limit]

        except Exception as e:
            print(f"Error finding nearby places: {e}")
            return []

    def _get_place_details(self, place_id: str) -> str:
        """
        Get detailed information for a specific place.

        Args:
            place_id: Google Places place ID

        Returns:
            Formatted address or empty string
        """
        if not self.gmaps or not place_id:
            return ""

        try:
            place_details = self.gmaps.place(
                place_id=place_id, fields=["formatted_address", "name", "types"]
            )

            result = place_details.get("result", {})
            return result.get("formatted_address", "")

        except Exception as e:
            print(f"Error getting place details: {e}")
            return ""

    def get_place_info(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get basic place information for a given location.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dictionary with basic place information
        """
        # Determine approximate location type based on coordinates
        location_type = self._determine_location_type(lat, lon)

        return {
            "formatted_address": f"Coordinates: {lat:.6f}, {lon:.6f}",
            "place_name": f"Location at {lat:.4f}°N, {lon:.4f}°W"
            if lon < 0
            else f"Location at {lat:.4f}°N, {lon:.4f}°E",
            "types": [location_type],
            "address_context": {
                "coordinates": f"{lat:.6f}, {lon:.6f}",
                "latitude": f"{lat:.6f}",
                "longitude": f"{lon:.6f}",
                "hemisphere": "Northern" if lat >= 0 else "Southern",
            },
            "geometry": {"location": {"lat": lat, "lng": lon}},
        }

    def _determine_location_type(self, lat: float, lon: float) -> str:
        """Determine approximate location type based on coordinates."""
        # Basic geographic analysis
        if abs(lat) > 60:
            return "polar_region"
        elif abs(lat) > 30:
            return "temperate_region"
        else:
            return "tropical_region"

    def analyze_with_gemini(
        self,
        place_info: Dict,
        user_search_context: str = "",
        nearby_places: List[Dict] = None,
    ) -> str:
        """
        Provide AI-powered analysis of the location and nearby places.

        Args:
            place_info: Place information
            user_search_context: What the user was searching for
            nearby_places: List of nearby places from database

        Returns:
            AI analysis of the location and nearby places
        """
        lat = place_info.get("geometry", {}).get("location", {}).get("lat", 0)
        lon = place_info.get("geometry", {}).get("location", {}).get("lng", 0)
        location_type = place_info.get("types", ["unknown"])[0]
        hemisphere = place_info.get("address_context", {}).get("hemisphere", "Unknown")

        # Start with basic geographic analysis
        analysis_parts = [
            f"📍 Geographic Location: {lat:.4f}°N, {lon:.4f}°W"
            if lon < 0
            else f"📍 Geographic Location: {lat:.4f}°N, {lon:.4f}°E",
            f"🌍 Hemisphere: {hemisphere}",
            f"🌡️ Climate Zone: {self._get_climate_zone(lat)}",
            f"🗺️ Region Type: {location_type.replace('_', ' ').title()}",
        ]

        # Add nearby places list (closest 5 within 1km)
        if nearby_places:
            analysis_parts.append("\n🏢 CLOSEST 5 PLACES FOUND WITHIN 1km:")
            for i, place in enumerate(nearby_places, 1):
                name = place.get("name", "Unknown")
                types = ", ".join(place.get("types", [])[:5])  # Show first 5 types
                address = place.get(
                    "formatted_address", place.get("vicinity", "No address")
                )
                rating = place.get("rating", "N/A")
                distance_m = place.get("distance_m", None)

                analysis_parts.append(f"  {i}. {name}")
                analysis_parts.append(f"     Types: {types}")
                if distance_m is not None:
                    if distance_m < 1000:
                        analysis_parts.append(f"     Distance: {distance_m:.0f}m")
                    else:
                        analysis_parts.append(
                            f"     Distance: {distance_m / 1000:.2f}km"
                        )
                if address:
                    analysis_parts.append(f"     Address: {address}")
                if rating != "N/A":
                    analysis_parts.append(f"     Rating: {rating}")
                analysis_parts.append("")  # Empty line between places

        # Add commodity value chain analysis if we have places and commodity context
        if user_search_context and nearby_places:
            analysis_parts.append(f"\n{'=' * 80}")
            analysis_parts.append(
                f"🎯 COMMODITY ANALYSIS: {user_search_context.upper()}"
            )
            analysis_parts.append(f"{'=' * 80}")
            analysis_parts.append(
                "Looking for FIRST POINTS OF AGGREGATION (processing facilities)"
            )
            analysis_parts.append(
                "where raw commodities are transported to and processed."
            )
            analysis_parts.append("")
            # AI-powered analysis of whether nearby places are processing facilities
            relevance_analysis = self._analyze_search_relevance(
                nearby_places, user_search_context
            )
            analysis_parts.append(relevance_analysis)
        elif user_search_context and not nearby_places:
            analysis_parts.append(
                f"\n⚠️ No places found within 1km for {user_search_context.upper()} analysis."
            )

        return "\n".join(analysis_parts)

    def _analyze_search_relevance(
        self, nearby_places: List[Dict], search_context: str
    ) -> str:
        """
        Analyze whether nearby places could be first points of aggregation for the commodity using Gemini AI.

        Args:
            nearby_places: List of nearby places (closest 5 within 1km, sorted by distance)
            search_context: Commodity name (e.g., coffee, palm oil, wood, etc.)

        Returns:
            AI analysis of whether places are processing facilities for the commodity
        """
        if not self.genai_client or not nearby_places:
            return f"🔍 Found {len(nearby_places)} nearby places within 1km. No AI analysis available."

        try:
            # Prepare detailed places information for Gemini
            places_info = []
            for i, place in enumerate(nearby_places, 1):
                name = place.get("name", "Unknown")
                types = ", ".join(place.get("types", []))  # Show all types
                address = place.get(
                    "formatted_address", place.get("vicinity", "No address")
                )
                rating = place.get("rating", "N/A")
                distance_m = place.get("distance_m", None)
                distance_str = (
                    f"{distance_m:.0f}m"
                    if distance_m and distance_m < 1000
                    else f"{distance_m / 1000:.2f}km"
                    if distance_m
                    else "unknown"
                )

                places_info.append(
                    f"{i}. {name} (Distance: {distance_str})\n   Types: {types}\n   Address: {address}\n   Rating: {rating}"
                )

            places_text = "\n".join(places_info)

            # Commodity-specific processing facility examples
            commodity_examples = {
                "coffee": "coffee cooperative, coffee processing plant, coffee mill, coffee warehouse, coffee exporter",
                "cocoa": "cocoa processing facility, cocoa cooperative, cocoa warehouse, chocolate factory, cocoa exporter",
                "palm oil": "palm oil mill, palm oil processing plant, palm oil refinery, palm oil warehouse, FFB collection point",
                "soy": "soy processing plant, soy mill, soy warehouse, soy crushing facility, soy exporter",
                "beef": "slaughterhouse, meat processing plant, cattle feedlot, meat packing facility, meat warehouse",
                "wood": "timber mill, lumber yard, sawmill, wood processing plant, forestry facility, wood warehouse",
                "rubber": "rubber processing plant, rubber factory, latex processing facility, rubber cooperative, rubber warehouse",
            }

            examples = commodity_examples.get(
                search_context.lower(),
                "processing facilities, mills, warehouses, factories",
            )

            # Create focused prompt for finding first points of aggregation
            prompt = f"""
You are analyzing locations for potential first points of aggregation in the {search_context} value chain. 

FIRST POINTS OF AGGREGATION are processing facilities where raw commodities are transported to and processed. Examples for {search_context} include: {examples}

The following are the 5 CLOSEST places found within 1km of the clicked location (sorted by distance):

{places_text}

TASK: Analyze these places to determine:
1. Are ANY of these places likely to be a first point of aggregation for {search_context}? (YES/NO)
2. If YES, which specific place(s) and WHY? (Consider name, types, and address context)
3. What type of facility is it? (e.g., mill, processing plant, cooperative, warehouse, factory)
4. Can you identify the COMPANY OWNERSHIP of the facility? 
   - If the place name contains a company name, identify it
   - If the address or context suggests company ownership, provide that information
   - If no clear company ownership can be determined from the available information, state "Company ownership not identifiable from available data"
5. CONFIDENCE LEVEL: (HIGH/MEDIUM/LOW) - How confident are you that this place is part of the {search_context} value chain?
6. OWNERSHIP CONFIDENCE: (HIGH/MEDIUM/LOW) - How confident are you in the company ownership identification (if provided)?

For {search_context.upper()} value chain, look for indicators like:
- Names containing: mill, processing, factory, plant, cooperative, warehouse, exporter, facility
- Place types: establishment, point_of_interest, store, food, or other industrial types
- Geographic context: rural/industrial areas are more likely than urban residential areas
- Address context: proximity to agricultural or industrial zones
- Company names: Often embedded in facility names (e.g., "PT. XYZ Palm Oil Mill", "ABC Coffee Cooperative", "XYZ Timber Ltd")

Provide a CLEAR and CONCISE assessment:
- List any places that COULD be first points of aggregation
- Explain WHY you think they might be (or not be) related to {search_context}
- Identify company ownership if discernible from name/address/context
- State your confidence level (HIGH/MEDIUM/LOW) for each potential match
- State your ownership confidence level if company ownership was identified

Format your response clearly with place names, reasoning, confidence levels, and company ownership information.
"""

            # Get Gemini AI analysis using new client
            response = self.genai_client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            ai_analysis = response.text.strip()

            return ai_analysis

        except Exception as e:
            print(f"Error in AI analysis: {e}")
            # Fallback to simple analysis
            return f"⚠️ Found {len(nearby_places)} nearby places within 1km. AI analysis unavailable due to error: {e}"

    def _get_climate_zone(self, lat: float) -> str:
        """Determine climate zone based on latitude."""
        abs_lat = abs(lat)
        if abs_lat < 23.5:
            return "Tropical"
        elif abs_lat < 35:
            return "Subtropical"
        elif abs_lat < 50:
            return "Temperate"
        elif abs_lat < 66.5:
            return "Subarctic"
        else:
            return "Arctic"

    def analyze_location(
        self, lat: float, lon: float, user_search_context: str = ""
    ) -> Dict:
        """
        Complete analysis pipeline: get place info, find nearby places, and provide AI analysis.
        Optimized for performance to not impact core labeling functionality.

        Args:
            lat: Latitude
            lon: Longitude
            user_search_context: What the user was searching for

        Returns:
            Dictionary with place info, nearby places, and AI analysis
        """
        # Performance check - skip if no API keys available
        if not self.gmaps or not self.genai_client:
            return {
                "success": False,
                "error": "Location analyzer not available (missing API keys)",
                "coordinates": (lat, lon),
                "timestamp": datetime.now().isoformat(),
            }

        try:
            # Get place information (fast, local operation)
            place_info = self.get_place_info(lat, lon)

            # Find the 5 closest places within 1km (optimized API call)
            nearby_places = self.find_nearby_places(lat, lon, limit=5)

            # Analyze location with nearby places context (only if we have places)
            if nearby_places:
                analysis = self.analyze_with_gemini(
                    place_info, user_search_context, nearby_places
                )
            else:
                analysis = self.analyze_with_gemini(place_info, user_search_context, [])

            return {
                "success": True,
                "place_info": place_info,
                "nearby_places": nearby_places,
                "gemini_analysis": analysis,
                "coordinates": (lat, lon),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Location analysis failed: {e}",
                "coordinates": (lat, lon),
                "timestamp": datetime.now().isoformat(),
            }


def create_location_analyzer(
    google_maps_api_key: str = None, gemini_api_key: str = None
) -> Optional[LocationAnalyzer]:
    """
    Create a LocationAnalyzer instance.

    Args:
        google_maps_api_key: Google Maps API key for Places API
        gemini_api_key: Google AI API key for Gemini analysis

    Returns:
        LocationAnalyzer instance, or None if dependencies are not available
    """
    if not GOOGLEMAPS_AVAILABLE or not GENAI_AVAILABLE:
        return None
    try:
        return LocationAnalyzer(google_maps_api_key, gemini_api_key)
    except Exception as e:
        print(f"Error creating LocationAnalyzer: {e}")
        return None
