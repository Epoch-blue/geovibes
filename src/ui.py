"""Map labeling interface and training utility functions for 
machine learning on top of satellite foundation model embeddings."""

import json
import os
import warnings
from datetime import datetime


import ee
import geopandas as gpd
import ipyleaflet as ipyl
from ipyleaflet import Map, Marker, basemaps, CircleMarker, LayerGroup, GeoJSON, DrawControl
from IPython.display import display
from ipywidgets import Button, FloatSlider, VBox, HBox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
import sklearn.metrics as metrics

from gee import get_s2_hsv_median, get_s2_rgb_median, get_ee_image_url, initialize_ee_with_credentials

warnings.simplefilter("ignore", category=FutureWarning)

initialize_ee_with_credentials()

# Get API keys from environment variables
MAPTILER_API_KEY = 'tBojLsXV1LWWhszN3ikf'
if not MAPTILER_API_KEY:
    MAPTILER_API_KEY = 'YOUR_MAPTILER_API_KEY'
    warnings.warn("MAPTILER_API_KEY environment variable not set. Using placeholder. Please set it for full functionality.")

MAPBOX_ACCESS_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN')
if not MAPBOX_ACCESS_TOKEN:
    MAPBOX_ACCESS_TOKEN = 'YOUR_MAPBOX_ACCESS_TOKEN'
    warnings.warn("MAPBOX_ACCESS_TOKEN environment variable not set. Using placeholder. Please set it for full functionality.")

BASEMAP_TILES = {
    'MAPTILER': f"https://api.maptiler.com/tiles/satellite-v2/{{z}}/{{x}}/{{y}}.jpg?key={MAPTILER_API_KEY}",
    # 'HUTCH_TILE': 'https://tiles.earthindex.ai/v1/tiles/sentinel2-temporal-mosaics/2023-01-01/2024-01-01/rgb/{z}/{x}/{y}.webp',
    'GOOGLE_HYBRID': 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    'MAPBOX': f"https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}.png?access_token={MAPBOX_ACCESS_TOKEN}"
}

class GeoLabeler:
    """An interactive Leaflet map for labeling geographic features relative to satellite image embedding tiles.
    
    Attributes: 
        gdf: A pandas GeoDataFrame whose columns are embedding feature values and a geometry
        map: A Leaflet map
        pos_ids, neg_ids: Lists of dataframe indices associated to pos / neg labeled points
        pos_layer, neg_layer, erase_layer, points: Leaflet map layers 
        select_val: 1/0/-100/2 to indicate pos/neg/erase/google maps label action
        execute_lable_point: Boolean flag for label_point() execution on map interaction
    
    External method: 
        update_layer: Add points to the map for visualization, without changing labels.
    
    """
    def __init__(
            self, geojson_path, mgrs_ids, start_date, end_date, imagery,
            duckdb_connection, baselayer_url=BASEMAP_TILES['MAPTILER'], **kwargs):
        print("Initializing GeoLabeler...")
        self.duckdb_connection = duckdb_connection
        self.current_basemap = 'MAPTILER'
        self.basemap_layer = ipyl.TileLayer(url=baselayer_url, no_wrap=True, name='basemap', 
                                       attribution=kwargs.get('attribution'))
        self.ee_boundary = ee.Geometry(shapely.geometry.mapping(
            gpd.read_file(geojson_path).geometry.iloc[0]))
        
        # Get map center from DuckDB
        load_spatial_query = """
        INSTALL spatial;
        LOAD spatial;
        """
        self.duckdb_connection.execute(load_spatial_query)

        # Get centroid of boundary from geopandas
        boundary_gdf = gpd.read_file(geojson_path)
        center_y, center_x = boundary_gdf.geometry.iloc[0].centroid.y, boundary_gdf.geometry.iloc[0].centroid.x
        
        self.map = Map(
            basemap=self.basemap_layer,
            center=(center_y, center_x), zoom=7, layout={'height':'600px'},
            scroll_wheel_zoom=True)

        hsv_median = get_s2_hsv_median(
            self.ee_boundary, start_date, end_date)

        hsv_url = get_ee_image_url(hsv_median, {
            'min': [0, 0, 0],
            'max': [1, 1, 1],
            'bands': ['hue', 'saturation', 'value']
        })
        BASEMAP_TILES['HSV_MEDIAN'] = hsv_url

        rgb_median = get_s2_rgb_median(
        self.ee_boundary, start_date, end_date, scale_factor=10000)

        rgb_url = get_ee_image_url(rgb_median, {
            'min': [0, 0, 0],
            'max': [0.25, 0.25, 0.25],
            'bands': ['B4', 'B3', 'B2']
        })
        BASEMAP_TILES['RGB_MEDIAN'] = rgb_url


        print("Adding controls...")
        self.pos_button = Button(description='Positive')
        self.neg_button = Button(description='Negative')
        self.erase_button = Button(description='Erase')
        self.google_maps_button = Button(description='Google Maps')
        self.toggle_mode_button = Button(description='Toggle Lasso Mode')
        self.toggle_basemap_button = Button(description=f'Basemap: {self.current_basemap}')
        self.save_button = Button(description='Save Dataset')
        self.pos_button.on_click(self.pos_click)
        self.neg_button.on_click(self.neg_click)
        self.erase_button.on_click(self.erase_click)
        self.google_maps_button.on_click(self.google_maps_click)
        self.toggle_mode_button.on_click(self.toggle_mode)
        self.toggle_basemap_button.on_click(self.toggle_basemap)
        #self.save_button.on_click(self.save_dataset)
        self.map.on_interaction(self.label_point)
        self.execute_label_point = True
        self.mgrs_ids = mgrs_ids
        self.select_val = -100 # Initialize to _erase_
        self.pos_ids = []
        self.neg_ids = []
        self.detection_gdf = None
        self.lasso_mode = False
        
        with open(geojson_path) as f:
            region_layer = ipyl.GeoJSON(
                    name="region",
                    data=json.load(f),
                    style={
                        'color': '#FAFAFA',
                        'opacity': 1,
                        'fillOpacity': 0,
                        'weight': 1
                    }
                )
        self.map.add_layer(region_layer)


        # layer to contain positive labeled points
        self.pos_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style={
                'color': 'green',
                'radius': 3,
                'fillColor': '#00FF00',
                'opacity': 1,
                'fillOpacity': 0.7,
                'weight': 1
            }
        )
        self.map.add_layer(self.pos_layer)

        # layer to contain negative labeled points
        self.neg_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style={
                'color': 'red',
                'radius': 3,
                'fillColor': '#FF0000',
                'opacity': 1,
                'fillOpacity': 0.7,
                'weight': 1
            }
        )
        self.map.add_layer(self.neg_layer)

        # erased points
        self.erase_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style={
                'color': 'white',
                'radius': 3,
                'fillColor': '#000000',
                'opacity': 1,
                'fillOpacity': 0.7,
                'weight': 1
            }
        )
        self.map.add_layer(self.erase_layer)
        
        # generic points layer for visualization
        self.points = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style={
                'color': 'black',
                'radius': 3,
                'fillColor': '#ffe014',
                'opacity': 1,
                'fillOpacity': 0.7,
                'weight': 1
            },
            hover_style={
                'fillColor': '#ffe014',
                'fillOpacity': 0.5
            }
        )
        self.map.add_layer(self.points)
        
        # Add DrawControl for lasso selection
        self.draw_control = DrawControl(
            polygon={"shapeOptions": {"color": "#6be5c3", "fillOpacity": 0.5}},
            polyline={},
            circle={},
            rectangle={},
            marker={},
            circlemarker={},
        )
        self.draw_control.polygon = {"shapeOptions": {"color": "#6be5c3"}}
        self.draw_control.on_draw(self.handle_draw)
        self.map.add_control(self.draw_control)
        self.draw_control.clear()

        display(VBox([
            self.map, 
            HBox([
                self.pos_button, 
                self.neg_button, 
                self.erase_button,
                self.google_maps_button,
                self.toggle_mode_button,
                self.toggle_basemap_button,
                self.save_button
            ])
        ]))
        
        self.query_vector = None
        self.detection_ids = []
        self.cached_embeddings = {}  # Add this to cache embeddings by point_id
        
    def pos_click(self, b):
        self.select_val = 1

    def neg_click(self, b):
        self.select_val = 0

    def erase_click(self, b):
        self.select_val = -100
        
    def google_maps_click(self, b):
        self.select_val = 2
        # Force single point mode when google maps is selected
        self.lasso_mode = False
        self.toggle_mode_button.description = 'Toggle Lasso Mode'
        self.draw_control.polygon = {}
        self.draw_control.clear()

    def toggle_mode(self, b):
        prev_select_val = self.select_val
        self.lasso_mode = not self.lasso_mode
        if self.lasso_mode:
            self.toggle_mode_button.description = 'Toggle Single Point Mode'
            self.draw_control.polygon = {"shapeOptions": {"color": "#6be5c3"}}
            # Restore previous selection mode
            self.select_val = prev_select_val
        else:
            self.toggle_mode_button.description = 'Toggle Lasso Mode'
            self.draw_control.polygon = {}
        self.draw_control.clear()

    def toggle_basemap(self, b):
        basemap_keys = list(BASEMAP_TILES.keys())
        current_idx = basemap_keys.index(self.current_basemap)
        next_idx = (current_idx + 1) % len(basemap_keys)
        self.current_basemap = basemap_keys[next_idx]
        
        # Update basemap layer
        self.basemap_layer.url = BASEMAP_TILES[self.current_basemap]
        self.toggle_basemap_button.description = f'Basemap: {self.current_basemap}'

    # def save_dataset(self, b):
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    #     # Save positive points
    #     if self.pos_ids:
    #         pos_gdf = self.gdf.loc[self.gdf['id'].isin(self.pos_ids)][["geometry"]]
    #         pos_gdf.to_file(f"positive_points_{timestamp}.geojson", driver="GeoJSON")
    #         print(f"Saved positive points to positive_points_{timestamp}.geojson")
    #     else:
    #         print("No positive points to save")
            
    #     # Save negative points
    #     if self.neg_ids:
    #         neg_gdf = self.gdf.loc[self.gdf['id'].isin(self.neg_ids)][["geometry"]]
    #         neg_gdf.to_file(f"negative_points_{timestamp}.geojson", driver="GeoJSON")
    #         print(f"Saved negative points to negative_points_{timestamp}.geojson")
    #     else:
    #         print("No negative points to save")

    def handle_draw(self, target, action, geo_json):
        if action != 'created':
            return
        
        # Get the polygon geometry from the drawn shape
        polygon_coords = geo_json['geometry']['coordinates'][0]
        polygon_wkt = f"POLYGON(({', '.join([f'{coord[0]} {coord[1]}' for coord in polygon_coords])}))"
        
        # Find points within the polygon using DuckDB and get their embeddings
        points_in_polygon_query = """
        SELECT id, embedding
        FROM geo_embeddings
        WHERE ST_Within(geometry, ST_GeomFromText(?))
        """
        
        points_inside = self.duckdb_connection.execute(points_in_polygon_query, [polygon_wkt]).df()
        
        print(f"Found {len(points_inside)} points inside polygon")
        for _, row in points_inside.iterrows():
            point_id = row['id']
            embedding = np.array(row['embedding'])
            
            # Cache the embedding
            self.cached_embeddings[point_id] = embedding
            
            if point_id in self.pos_ids:
                self.pos_ids.remove(point_id)
            if point_id in self.neg_ids:
                self.neg_ids.remove(point_id)
            
            if self.select_val == 1:
                self.pos_ids.append(point_id)
            elif self.select_val == 0:
                self.neg_ids.append(point_id)
        
        self.update_layers()
        self.update_query_vector()
        self.draw_control.clear()

    def update_layers(self):
        if self.pos_ids:
            pos_query = """
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings 
            WHERE id IN ({})
            """.format(','.join([f"'{pid}'" for pid in self.pos_ids]))
            pos_results = self.duckdb_connection.execute(pos_query).df()
            pos_geojson = {"type": "FeatureCollection", "features": []}
            for _, row in pos_results.iterrows():
                pos_geojson["features"].append({
                    "type": "Feature", 
                    "geometry": json.loads(row['geometry']),
                    "properties": {}
                })
            self.pos_layer.data = pos_geojson
        else:
            self.pos_layer.data = {"type": "FeatureCollection", "features": []}
        
        if self.neg_ids:
            neg_query = """
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings 
            WHERE id IN ({})
            """.format(','.join([f"'{nid}'" for nid in self.neg_ids]))
            neg_results = self.duckdb_connection.execute(neg_query).df()
            neg_geojson = {"type": "FeatureCollection", "features": []}
            for _, row in neg_results.iterrows():
                neg_geojson["features"].append({
                    "type": "Feature", 
                    "geometry": json.loads(row['geometry']),
                    "properties": {}
                })
            self.neg_layer.data = neg_geojson
        else:
            self.neg_layer.data = {"type": "FeatureCollection", "features": []}

    def label_point(self, **kwargs):
        """Assign a label and map layer to a clicked map point."""
        if not self.execute_label_point or self.lasso_mode:
            return
        
        action = kwargs.get('type') 
        if action not in ['click']:
            return
                 
        lat, lon = kwargs.get('coordinates')
        
        if self.select_val == 2:
            import webbrowser
            url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            webbrowser.open(url, new=2, autoraise=True)
            return
        
        # Pull both geometry info AND embedding in one query
        sql = """
        SELECT  g.id,
                ST_AsText(g.geometry) AS wkt,
                ST_Distance(geometry, ST_Point(?, ?)) AS dist_m,
                g.embedding
        FROM    geo_embeddings g
        ORDER BY dist_m
        LIMIT   1
        """
        
        nearest_result = self.duckdb_connection.execute(sql, [lon, lat]).df()
        
        if nearest_result.empty:
            return
        
        point_id = nearest_result.iloc[0]['id']
        embedding = nearest_result.iloc[0]['embedding']
        
        # Cache the embedding for later use
        self.cached_embeddings[point_id] = np.array(embedding)
        
        if point_id in self.pos_ids:
            self.pos_ids.remove(point_id)
        if point_id in self.neg_ids:
            self.neg_ids.remove(point_id)
            
        if self.select_val == 1:
            self.pos_ids.append(point_id)
        elif self.select_val == 0:
            self.neg_ids.append(point_id)
        else:
            # For erase mode, get the point geometry from DuckDB
            erase_query = """
            SELECT ST_AsGeoJSON(geometry) as geometry
            FROM geo_embeddings 
            WHERE id = ?
            """
            erase_result = self.duckdb_connection.execute(erase_query, [point_id]).fetchone()
            if erase_result:
                erase_geojson = {
                    "type": "FeatureCollection", 
                    "features": [{
                        "type": "Feature", 
                        "geometry": json.loads(erase_result[0]),
                        "properties": {}
                    }]
                }
                self.erase_layer.data = erase_geojson
        
        self.update_layers()
        self.update_query_vector()  # This will now use cached embeddings

    def update_layer(self, layer, new_data):
        """Add points to the map for visualization, without changing labels."""
        self.execute_label_point = False
        layer.data = new_data
        self.execute_label_point = True

    def update_query_vector(self):
        """Update the query vector based on current positive and negative labels."""
        if not self.pos_ids:
            self.query_vector = None
            return
        
        # Use cached embeddings instead of querying database
        pos_embeddings = []
        missing_pos_ids = []
        
        for pid in self.pos_ids:
            if pid in self.cached_embeddings:
                pos_embeddings.append(self.cached_embeddings[pid])
            else:
                missing_pos_ids.append(pid)
        
        # If we have missing embeddings, fetch them
        if missing_pos_ids:
            query = """
            SELECT id, embedding
            FROM geo_embeddings 
            WHERE id IN ({})
            """.format(','.join([f"'{pid}'" for pid in missing_pos_ids]))
            
            missing_results = self.duckdb_connection.execute(query).df()
            for _, row in missing_results.iterrows():
                embedding = np.array(row['embedding'])
                self.cached_embeddings[row['id']] = embedding
                pos_embeddings.append(embedding)
        
        if not pos_embeddings:
            self.query_vector = None
            return
        
        pos_vec = np.mean(pos_embeddings, axis=0)
        
        # Handle negative embeddings
        neg_embeddings = []
        missing_neg_ids = []
        
        for nid in self.neg_ids:
            if nid in self.cached_embeddings:
                neg_embeddings.append(self.cached_embeddings[nid])
            else:
                missing_neg_ids.append(nid)
        
        # If we have missing negative embeddings, fetch them
        if missing_neg_ids:
            query = """
            SELECT id, embedding
            FROM geo_embeddings 
            WHERE id IN ({})
            """.format(','.join([f"'{nid}'" for nid in missing_neg_ids]))
            
            missing_results = self.duckdb_connection.execute(query).df()
            for _, row in missing_results.iterrows():
                embedding = np.array(row['embedding'])
                self.cached_embeddings[row['id']] = embedding
                neg_embeddings.append(embedding)
        
        if neg_embeddings:
            neg_vec = np.mean(neg_embeddings, axis=0)
        else:
            neg_vec = np.zeros_like(pos_vec)
        
        # Default query vector math
        self.query_vector = 2 * pos_vec - neg_vec
        print(f"Updated query vector from {len(self.pos_ids)} positive and {len(self.neg_ids)} negative labels")
