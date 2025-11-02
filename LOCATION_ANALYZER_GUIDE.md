# Location Analyzer with Google Geocoding API and Gemini AI

This feature adds AI-powered location analysis to GeoVibes, allowing users to get contextual information about clicked locations using Google Geocoding API and Gemini AI. The implementation leverages the new building footprint capabilities of the Geocoding API.

## Setup

### 1. Install Dependencies
```bash
pip install google-generativeai googlemaps
```

### 2. Set Environment Variables
```bash
export GOOGLE_MAPS_API_KEY="xxx"
export GEMINI_API_KEY="xxx"
```

### 3. Get API Keys

**Google Maps API Key:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable "Geocoding API" (includes building footprint data)
4. Create credentials → API Key
5. Restrict the key to your domains (recommended)

**Gemini API Key:**
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key"
3. Create a new API key
4. Copy the key for your environment variable

## Usage

### Basic Usage
```python
from geovibes import GeoVibes

# Initialize GeoVibes (location analyzer will be auto-initialized if API keys are set)
app = GeoVibes.create(
    duckdb_path="path/to/your/database.db",
    verbose=True
)

# After performing a search and clicking on a result:
app.analyze_location(lat=40.7128, lon=-74.0060)
```

### Programmatic Usage
```python
from geovibes.ui.location_analyzer import LocationAnalyzer

# Create analyzer directly
analyzer = LocationAnalyzer(google_maps_api_key, gemini_api_key)

# Analyze a location
result = analyzer.analyze_location(
    lat=40.7128, 
    lon=-74.0060, 
    user_search_context="hospitals"
)

if result['success']:
    print("Place:", result['place_info']['name'])
    print("Analysis:", result['gemini_analysis'])
```

## Features

### 1. Google Geocoding Integration
- **Reverse geocoding**: Get place names and addresses from coordinates
- **Building footprints**: Extract building/ground footprint data when available
- **Address components**: Structured location data (neighborhood, city, state, country)
- **Location types**: Premise, establishment, point of interest classification

### 2. Gemini AI Analysis
- **Contextual understanding**: What type of facility/feature
- **Search validation**: Does it match what user was looking for?
- **Confidence assessment**: High/medium/low confidence levels
- **Detailed insights**: Specific characteristics and features

### 3. Smart Integration
- **Search context awareness**: Uses your search terms for better analysis
- **Error handling**: Graceful fallbacks when APIs are unavailable
- **Memory efficient**: No data storage, just real-time analysis

## Example Output

```
📍 Location Analysis

Place: Mount Sinai Hospital
Address: 1 Gustave L Levy Pl, New York, NY 10029, USA
Type: hospital, establishment, health, point_of_interest

AI Analysis:
This is a major hospital and medical center. It's a healthcare facility that provides medical services, likely matching your search for "hospitals". The location is a well-known medical institution with high confidence that this matches your search criteria.
```

## Use Cases

### 1. Search Validation
- **User searches for "hospitals"** → Click result → AI confirms it's actually a hospital
- **User searches for "industrial facilities"** → AI identifies specific type of industrial site
- **User searches for "agricultural areas"** → AI confirms farming practices or crop types

### 2. Context Understanding
- **Unknown locations**: Get place names and business information
- **Facility types**: Understand what type of establishment was found
- **Geographic context**: Learn about the area and its characteristics

### 3. Quality Control
- **False positives**: Identify when search results don't match intent
- **Result validation**: Confirm similarity search accuracy
- **Learning tool**: Help users understand what embeddings represent

## Troubleshooting

### Common Issues

**"Location analyzer not available"**
- Check that both API keys are set correctly
- Verify API keys have proper permissions
- Ensure dependencies are installed

**"No place information found"**
- Location might be in a remote area
- Try increasing the search radius
- Check if coordinates are valid

**"Error analyzing location"**
- Check API quotas and billing
- Verify network connectivity
- Check API key restrictions

### API Limits
- **Google Places API**: 1000 requests/day (free tier)
- **Gemini API**: 15 requests/minute (free tier)
- Consider upgrading for higher limits

## Advanced Configuration

### Custom Search Radius
```python
# Modify the analyzer to use different search radius
analyzer = LocationAnalyzer(api_key, gemini_key)
analyzer.get_place_info(lat, lon, radius=100)  # 100m radius
```

### Custom Prompts
```python
# Modify the Gemini prompt for specific use cases
def custom_analyze_with_gemini(self, place_info, user_context):
    prompt = f"""
    You are a geospatial analyst. Analyze this location:
    {place_info}
    
    User context: {user_context}
    
    Provide: facility type, confidence, and recommendations.
    """
    # ... rest of implementation
```

## Integration with GeoVibes

The location analyzer is automatically integrated into the GeoVibes interface:

1. **Automatic initialization**: Set up when GeoVibes starts (if API keys available)
2. **Click integration**: Can be called when users click on search results
3. **Status display**: Results shown in the operation status area
4. **Error handling**: Graceful fallbacks when services unavailable

This creates a seamless experience where users can validate their search results with AI-powered insights about the actual places they find.
