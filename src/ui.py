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
from ipywidgets import Button, FloatSlider, VBox, HBox, IntSlider, Label, Layout, HTML
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
    'HUTCH_TILE': 'https://tiles.earthindex.ai/v2/tiles/sentinel2-temporal-mosaics/2023-01-01/2024-01-01/rgb/{z}/{x}/{y}.webp',
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
        
        # Button instances
        self.pos_button = Button(description='Positive')
        self.neg_button = Button(description='Negative')
        self.erase_button = Button(description='Erase')
        self.google_maps_button = Button(description='Maps')
        self.toggle_mode_button = Button(description='Lasso')
        self.toggle_basemap_button = Button(description=f'🛰 {self.current_basemap}')
        self.search_button = Button(description='Search')
        self.save_button = Button(description='💾')

        # Initialize styles and layouts
        self._initialize_button_styles_and_layouts()

        # Neighbors slider
        self.neighbors_slider = IntSlider(
            value=1000,
            min=100,
            max=10000,
            step=100,
            description='Neighbors:',
            style={'description_width': 'initial'},
            layout=Layout(width='300px')
        )
        
        # Set up click handlers
        self.pos_button.on_click(self.pos_click)
        self.neg_button.on_click(self.neg_click)
        self.erase_button.on_click(self.erase_click)
        self.google_maps_button.on_click(self.google_maps_click)
        self.toggle_mode_button.on_click(self.toggle_mode)
        self.toggle_basemap_button.on_click(self.toggle_basemap)
        self.search_button.on_click(self.search_click)
        self.save_button.on_click(self.save_dataset)
        
        self.map.on_interaction(self.label_point)
        self.execute_label_point = True
        self.mgrs_ids = mgrs_ids
        self.select_val = -100 # Initialize to _erase_
        self.pos_ids = []
        self.neg_ids = []
        self.detection_gdf = None
        self.lasso_mode = False

        # Update button styles to reflect initial state
        self._update_active_button_styles()
        
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
            HTML("<hr style='margin: 10px 0;'>"),
            HBox([
                VBox([
                    Label("Labeling Controls:", style={'font_weight': 'bold'}),
                    HBox([
                        self.pos_button, 
                        self.neg_button, 
                        self.erase_button,
                        self.google_maps_button
                    ], layout=Layout(margin='5px 0'))
                ]),
                VBox([
                    Label("Mode & View:", style={'font_weight': 'bold'}),
                    HBox([
                        self.toggle_mode_button,
                        self.toggle_basemap_button
                    ], layout=Layout(margin='5px 0'))
                ]),
                VBox([
                    Label("Search & Save:", style={'font_weight': 'bold'}),
                    HBox([
                        self.search_button,
                        self.save_button, # Save button is here
                        self.neighbors_slider
                    ], layout=Layout(margin='5px 0'))
                ])
            ], layout=Layout(justify_content='space-between', margin='10px'))
        ]))
        
        self.query_vector = None
        self.detection_ids = []
        self.cached_embeddings = {}
        self.detections_with_embeddings = None  # GeoDataFrame cache
        
    def _initialize_button_styles_and_layouts(self):
        # Define base layouts and styles with rounded corners, shadow, and grey color
        self.base_layout_config = {
            'height': '35px', 
            'border_radius': '8px', 
            'margin': '0 3px', 
            'padding': '0 8px', 
            'border': '1px solid #cccccc', 
            'width': '120px'  # Make buttons wider
        }
        self.active_layout_config = self.base_layout_config.copy()
        self.active_layout_config['border'] = '2px solid #007bff'
        
        # Grey background with shadow
        self.base_style = {
            'button_color': '#f0f0f0', 
            'font_weight': 'normal'
        }
        self.active_style = {
            'button_color': '#e0e0e0', 
            'font_weight': 'bold'
        }

        # Save button: keep icon only and smaller width
        self.save_button_layout_config = self.base_layout_config.copy()
        self.save_button_layout_config['width'] = '45px'
        
        # Basemap button: keep current width (don't make wider)
        self.basemap_button_layout_config = self.base_layout_config.copy()
        self.basemap_button_layout_config['width'] = '140px'

        self.all_buttons = [
            self.pos_button, self.neg_button, self.erase_button, self.google_maps_button,
            self.toggle_mode_button, self.toggle_basemap_button, self.search_button, self.save_button
        ]
        
        for btn in self.all_buttons:
            current_layout_config = self.base_layout_config.copy()
            
            if btn == self.save_button:
                current_layout_config = self.save_button_layout_config.copy()
            elif btn == self.toggle_basemap_button:
                current_layout_config = self.basemap_button_layout_config.copy()
            
            btn.layout = Layout(**current_layout_config)
            btn.style = self.base_style.copy()
            
            # Add CSS for shadow effect
            btn.add_class('custom-button')
        
        # Add custom CSS for shadow effect
        from IPython.display import HTML, display
        custom_css = HTML("""
        <style>
        .custom-button {
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        .custom-button:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        }
        </style>
        """)
        display(custom_css)

    def _update_active_button_styles(self):
        # Reset all potentially active buttons to base state first
        label_buttons = [self.pos_button, self.neg_button, self.erase_button, self.google_maps_button]
        
        for btn in label_buttons:
            btn.layout.border = self.base_layout_config['border']
            btn.style = self.base_style.copy()

        self.toggle_mode_button.layout.border = self.base_layout_config['border']
        self.toggle_mode_button.style = self.base_style.copy()

        # Apply active state to the selected labeling button
        active_button = None
        if self.select_val == 1:
            active_button = self.pos_button
        elif self.select_val == 0:
            active_button = self.neg_button
        elif self.select_val == -100: 
            active_button = self.erase_button
        elif self.select_val == 2: 
            active_button = self.google_maps_button
        
        if active_button:
            active_button.layout.border = self.active_layout_config['border']
            active_button.style = self.active_style.copy()

        # Apply active state to toggle_mode_button if lasso is active
        if self.lasso_mode:
            self.toggle_mode_button.layout.border = self.active_layout_config['border']
            self.toggle_mode_button.style = self.active_style.copy()
            self.toggle_mode_button.description = 'Single' 
        else:
            self.toggle_mode_button.description = 'Lasso'
            # If single point is the default mode, it could also be highlighted.
            # For now, only lasso mode gets the distinct highlight on the toggle button itself.

    def pos_click(self, b):
        self.select_val = 1
        self._update_active_button_styles()

    def neg_click(self, b):
        self.select_val = 0
        self._update_active_button_styles()

    def erase_click(self, b):
        self.select_val = -100
        self._update_active_button_styles()
        
    def google_maps_click(self, b):
        self.select_val = 2
        if self.lasso_mode: # If currently in lasso mode, turn it off
            self.lasso_mode = False
            self.draw_control.polygon = {} 
            self.draw_control.clear()
        self._update_active_button_styles()

    def toggle_mode(self, b):
        prev_select_val = self.select_val 
        self.lasso_mode = not self.lasso_mode
        
        if self.lasso_mode:
            self.toggle_mode_button.description = 'Single'
            self.draw_control.polygon = {"shapeOptions": {"color": "#6be5c3"}}
            if self.select_val == 2: 
                self.select_val = 1 
        else: 
            self.toggle_mode_button.description = 'Lasso'
            self.draw_control.polygon = {}
        
        self.draw_control.clear()
        self._update_active_button_styles()

    def toggle_basemap(self, b):
        basemap_keys = list(BASEMAP_TILES.keys())
        current_idx = basemap_keys.index(self.current_basemap)
        next_idx = (current_idx + 1) % len(basemap_keys)
        self.current_basemap = basemap_keys[next_idx]
        
        # Update basemap layer
        self.basemap_layer.url = BASEMAP_TILES[self.current_basemap]
        self.toggle_basemap_button.description = f'🛰 {self.current_basemap}'
        self._update_active_button_styles()

    def search_click(self, b):
        """Perform similarity search based on current query vector."""
        if self.query_vector is None:
            print("⚠️ No query vector available. Please add some positive labels first.")
            return
        
        n_neighbors = self.neighbors_slider.value
        
        # Convert query vector to the format needed for DuckDB
        query_vec = self.query_vector.tolist()
        
        sql = """
        WITH query(vec) AS (SELECT CAST(? AS FLOAT[384]))
        SELECT  g.id,
                g.embedding,
                ST_AsGeoJSON(g.geometry) AS geometry_json,
                ST_AsText(g.geometry) AS geometry_wkt,
                array_distance(g.embedding, q.vec) AS distance
        FROM    geo_embeddings AS g, query AS q
        ORDER BY distance
        LIMIT ?;
        """
        
        print(f"🔍 Searching for {n_neighbors} similar points...")
        search_results = self.duckdb_connection.execute(sql, [query_vec, n_neighbors]).df()
        
        # Filter out any IDs that are already in positive labels
        search_results_filtered = search_results[~search_results['id'].isin(self.pos_ids)]
        print(search_results_filtered.head())
        # Create GeoDataFrame with embeddings for caching
        # Use shapely.wkb.loads for WKB geometry from DuckDB
        geometries = [shapely.wkt.loads(row['geometry_wkt']) if row['geometry_wkt'] else None
                     for _, row in search_results_filtered.iterrows()]
        
        self.detections_with_embeddings = gpd.GeoDataFrame({
            'id': search_results_filtered['id'].values,
            'embedding': search_results_filtered['embedding'].values,
            'distance': search_results_filtered['distance'].values,
            'geometry': geometries
        })
        
        # Create GeoJSON for map display
        detections_geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for _, row in search_results_filtered.iterrows():
            detections_geojson["features"].append({
                "type": "Feature",
                "geometry": json.loads(row['geometry_json']),
                "properties": {"id": row['id'], "distance": row['distance']}
            })
        
        # Update the map
        self.update_layer(self.points, detections_geojson)
        
        print(f"✅ Found {len(detections_geojson['features'])} similar points")

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
        
        clicked_point = Point(lon, lat)
        point_id = None
        embedding = None
        
        # First check if we have cached detections
        if self.detections_with_embeddings is not None and len(self.detections_with_embeddings) > 0:
            # Find nearest point in cached detections
            distances = self.detections_with_embeddings.geometry.distance(clicked_point)
            nearest_idx = distances.idxmin()
            
            # Use a threshold to ensure we're clicking on an actual point
            if distances[nearest_idx] < 0.001:  # Adjust threshold as needed
                nearest_detection = self.detections_with_embeddings.loc[nearest_idx]
                point_id = nearest_detection['id']
                embedding = nearest_detection['embedding']
        
        # If not found in cache, query the database
        if point_id is None:
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
        
        # Update labels
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
        self.update_query_vector()

    def update_layer(self, layer, geojson_data):
        """Update a specific layer with new GeoJSON data."""
        layer.data = geojson_data

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

        self.update_query_vector()

    def handle_draw(self, target, action, geo_json):
        if action != 'created':
            return
        
        # Get the polygon geometry from the drawn shape and convert to shapely Polygon
        polygon_coords = geo_json['geometry']['coordinates'][0]
        polygon = shapely.geometry.Polygon(polygon_coords)
        
        points_to_label = []
        
        # First check cached detections
        if self.detections_with_embeddings is not None and len(self.detections_with_embeddings) > 0:
            # Find points within polygon from cached detections
            within_mask = self.detections_with_embeddings.geometry.within(polygon)
            cached_points = self.detections_with_embeddings[within_mask]
            
            for _, row in cached_points.iterrows():
                points_to_label.append({
                    'id': row['id'],
                    'embedding': row['embedding']
                })
        
        # If no cached results or need more points, query the database
        if len(points_to_label) == 0:
            polygon_wkt = polygon.wkt
            
            points_in_polygon_query = f"""
            SELECT id, embedding
            FROM geo_embeddings
            WHERE ST_Within(geometry, ST_GeomFromText('{polygon_wkt}'))
            """
            
            points_inside = self.duckdb_connection.execute(points_in_polygon_query).df()
            
            for _, row in points_inside.iterrows():
                points_to_label.append({
                    'id': row['id'],
                    'embedding': np.array(row['embedding'])
                })
        
        print(f"Found {len(points_to_label)} points inside polygon")
        
        # Label the points
        for point in points_to_label:
            point_id = point['id']
            embedding = point['embedding']
            
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

    def save_dataset(self, b):
        """Save labeled points with embeddings to a GeoJSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if we have any labels to save
        if not self.pos_ids and not self.neg_ids:
            print("⚠️ No labeled points to save.")
            return
        
        print("💾 Saving dataset...")
        
        # Combine all labeled IDs
        all_labeled_ids = list(set(self.pos_ids + self.neg_ids))
        
        if not all_labeled_ids:
            print("⚠️ No valid labels to save.")
            return
        
        # Query database for all labeled points with their geometries and embeddings
        query = """
        SELECT 
            id,
            ST_AsText(geometry) AS wkt,
            ST_AsGeoJSON(geometry) AS geometry_json,
            embedding
        FROM geo_embeddings 
        WHERE id IN ({})
        """.format(','.join([f"'{lid}'" for lid in all_labeled_ids]))
        
        results = self.duckdb_connection.execute(query).df()
        
        if results.empty:
            print("⚠️ Could not retrieve data for labeled points.")
            return
        
        # Create lists to store the data
        features = []
        
        # Process each result
        for _, row in results.iterrows():
            point_id = row['id']
            
            # Determine label (1 for positive, 0 for negative)
            if point_id in self.pos_ids:
                label = 1
            elif point_id in self.neg_ids:
                label = 0
            else:
                continue  # Skip if somehow not in either list
            
            # Get embedding (from cache or from query result)
            if point_id in self.cached_embeddings:
                embedding = self.cached_embeddings[point_id]
            else:
                embedding = np.array(row['embedding'])
            
            # Create feature with properties including label and embedding
            feature = {
                "type": "Feature",
                "geometry": json.loads(row['geometry_json']),
                "properties": {
                    "id": point_id,
                    "label": label,
                    "embedding": embedding.tolist()  # Convert numpy array to list for JSON serialization
                }
            }
            features.append(feature)
        
        # Create GeoJSON structure
        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "timestamp": timestamp,
                "total_points": len(features),
                "positive_points": len([f for f in features if f['properties']['label'] == 1]),
                "negative_points": len([f for f in features if f['properties']['label'] == 0]),
                "embedding_dimension": len(features[0]['properties']['embedding']) if features else 0
            }
        }
        
        # Save to file
        filename = f"labeled_dataset_{timestamp}.geojson"
        
        try:
            with open(filename, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            # Create summary
            pos_count = len([f for f in features if f['properties']['label'] == 1])
            neg_count = len([f for f in features if f['properties']['label'] == 0])
            
            print(f"✅ Dataset saved successfully!")
            print(f"📄 Filename: {filename}")
            print(f"📊 Summary:")
            print(f"   - Total points: {len(features)}")
            print(f"   - Positive labels: {pos_count}")
            print(f"   - Negative labels: {neg_count}")
            print(f"   - Embedding dimension: {len(features[0]['properties']['embedding']) if features else 0}")
            
            # Optional: Also save a separate CSV with just IDs and labels for easier processing
            labels_df = pd.DataFrame([
                {'id': f['properties']['id'], 'label': f['properties']['label']} 
                for f in features
            ])
            csv_filename = f"labeled_dataset_{timestamp}_labels.csv"
            labels_df.to_csv(csv_filename, index=False)
            print(f"📄 Also saved labels CSV: {csv_filename}")
            
        except Exception as e:
            print(f"❌ Error saving dataset: {str(e)}")

    def load_dataset(self, filename):
        """Load a previously saved labeled dataset."""
        try:
            with open(filename, 'r') as f:
                geojson_data = json.load(f)
            
            # Clear current labels
            self.pos_ids = []
            self.neg_ids = []
            self.cached_embeddings = {}
            
            # Process features
            for feature in geojson_data['features']:
                point_id = feature['properties']['id']
                label = feature['properties']['label']
                embedding = np.array(feature['properties']['embedding'])
                
                # Cache the embedding
                self.cached_embeddings[point_id] = embedding
                
                # Add to appropriate list
                if label == 1:
                    self.pos_ids.append(point_id)
                elif label == 0:
                    self.neg_ids.append(point_id)
            
            # Update visualization
            self.update_layers()
            self.update_query_vector()
            
            # Print summary
            metadata = geojson_data.get('metadata', {})
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Summary:")
            print(f"   - Total points: {metadata.get('total_points', len(geojson_data['features']))}")
            print(f"   - Positive labels: {len(self.pos_ids)}")
            print(f"   - Negative labels: {len(self.neg_ids)}")
            print(f"   - Saved on: {metadata.get('timestamp', 'Unknown')}")
            
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
