"""Map labeling interface and training utility functions for 
machine learning on top of satellite foundation model embeddings."""

import json
import warnings
from datetime import datetime

import duckdb
import ee
import geopandas as gpd
import ipyleaflet as ipyl
from ipyleaflet import Map, DrawControl
from IPython.display import display
from ipywidgets import Button, VBox, HBox, IntSlider, Label, Layout, HTML, ToggleButtons, Accordion, FileUpload
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
import webbrowser

from .ee_tools import get_s2_ndvi_median, get_s2_ndwi_median, get_ee_image_url, initialize_ee_with_credentials
from .ui_config import UIConstants, BasemapConfig, GeoLabelerConfig, DatabaseConstants, LayerStyles

warnings.simplefilter("ignore", category=FutureWarning)


EE_AVAILABLE = initialize_ee_with_credentials()

# Validate MapTiler API key
if not BasemapConfig.MAPTILER_API_KEY:
    warnings.warn("MAPTILER_API_KEY environment variable not set. Please create a .env file with your MapTiler API key.")


class GeoVibes:
    """An interactive Leaflet map for labeling geographic features relative to satellite image embedding tiles.
    
    Attributes: 
        gdf: A pandas GeoDataFrame whose columns are embedding feature values and a geometry
        map: A Leaflet map
        pos_ids, neg_ids: Lists of dataframe indices associated to pos / neg labeled points
        pos_layer, neg_layer, erase_layer, points: Leaflet map layers 
        select_val: 1/0/-100 to indicate pos/neg/erase label action
        execute_lable_point: Boolean flag for label_point() execution on map interaction
    
    External method: 
        update_layer: Add points to the map for visualization, without changing labels.
    
    """
    
    @classmethod
    def from_config(cls, config_path, **kwargs):
        """Create a GeoLabeler instance from a configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            **kwargs: Additional keyword arguments to override config values
        
        Returns:
            GeoLabeler instance
        """
        return cls(config_path=config_path, **kwargs)
    def __init__(
            self, geojson_path=None, start_date=None, end_date=None,
            duckdb_connection=None, duckdb_path=None, config=None, config_path=None,
            baselayer_url=None, **kwargs):
        print("Initializing GeoLabeler...")
        
        # Handle configuration loading using new config system
        if config_path is not None:
            self.config = GeoLabelerConfig.from_file(config_path)
            self.config.validate()
        elif config is not None:
            self.config = GeoLabelerConfig.from_dict(config)
            self.config.validate()
        else:
            # Create config from individual parameters
            if geojson_path is None or start_date is None or end_date is None:
                raise ValueError("Required parameters missing. Provide either config_path, config dict, or individual parameters.")
            self.config = GeoLabelerConfig(
                duckdb_path=duckdb_path,
                boundary_path=geojson_path,
                start_date=start_date,
                end_date=end_date
            )
            self.config.validate()
        
        # Set default baselayer URL if not provided
        if baselayer_url is None:
            baselayer_url = BasemapConfig.BASEMAP_TILES['MAPTILER']
        
        # Handle duckdb connection
        if duckdb_connection is None:
            self.duckdb_connection = duckdb.connect(self.config.duckdb_path)
            self._owns_connection = True
        else:
            self.duckdb_connection = duckdb_connection
            self._owns_connection = False
        self.current_basemap = 'MAPTILER'
        self.basemap_layer = ipyl.TileLayer(url=baselayer_url, no_wrap=True, name='basemap', 
                                       attribution=BasemapConfig.MAPTILER_ATTRIBUTION)
        
        # Setup Earth Engine boundary if available
        if EE_AVAILABLE:
            try:
                self.ee_boundary = ee.Geometry(shapely.geometry.mapping(
                    gpd.read_file(self.config.boundary_path).union_all()))
            except Exception as e:
                print(f"⚠️  Failed to create Earth Engine boundary: {e}")
                print("⚠️  NDVI/NDWI basemaps will be unavailable")
                self.ee_boundary = None
        else:
            self.ee_boundary = None
        
        # Setup spatial extension in DuckDB
        self.duckdb_connection.execute(DatabaseConstants.SPATIAL_SETUP_QUERY)

        # Get centroid of boundary from geopandas
        boundary_gdf = gpd.read_file(self.config.boundary_path)
        center_y, center_x = boundary_gdf.geometry.iloc[0].centroid.y, boundary_gdf.geometry.iloc[0].centroid.x
        
        # Build map
        self.map = self._build_map(center_y, center_x)

        # Add Earth Engine basemap options (if available)
        self._setup_ee_basemaps()

        print("Building UI...")
        
        # Initialize state
        self.current_label = 'Positive'
        self.execute_label_point = True
        self.select_val = UIConstants.POSITIVE_LABEL  # Initialize to positive
        self.pos_ids = []
        self.neg_ids = []
        self.detection_gdf = None
        self.lasso_mode = False
        self.query_vector = None
        self.detection_ids = []
        self.cached_embeddings = {}
        self.detections_with_embeddings = None
        
        # Build UI
        self.side_panel, self.ui_widgets = self._build_side_panel()
        
        # Add layers to map
        self._add_map_layers()
        
        # Add DrawControl
        self._setup_draw_control()
        
        # Wire events
        self._wire_events()
        
        # Add legend
        self.legend = HTML(value=f"""
            <div style='background: white; padding: 5px; border-radius: 5px; opacity: 0.8;'>
                <span style='color: {UIConstants.POS_COLOR}; font-weight: bold;'>🔵 Positive</span> | 
                <span style='color: {UIConstants.NEG_COLOR}; font-weight: bold;'>🟠 Negative</span>
            </div>
        """)
        
        # Add status bar
        self.status_bar = HTML(value="Ready")
        
        # Create main layout
        map_with_overlays = VBox([
            self.map,
            HBox([self.legend, self.status_bar], 
                 layout=Layout(justify_content='space-between', padding='5px'))
        ], layout=Layout(flex='1 1 auto'))
        
        self.main_layout = HBox([
            self.side_panel,
            map_with_overlays
        ], layout=Layout(height=UIConstants.DEFAULT_HEIGHT, width='100%'))
        
        display(self.main_layout)


    def _setup_ee_basemaps(self):
        """Set up Earth Engine basemaps (NDVI and NDWI) if available."""
        # Create a copy of the base basemap tiles
        self.basemap_tiles = BasemapConfig.BASEMAP_TILES.copy()
        
        # Only add Earth Engine basemaps if EE is available and boundary is set
        if EE_AVAILABLE and self.ee_boundary is not None:
            try:
                print("🛰️ Setting up Earth Engine basemaps (NDVI and NDWI)...")
                
                # Add NDVI basemap
                ndvi_median = get_s2_ndvi_median(
                    self.ee_boundary, self.config.start_date, self.config.end_date)
                ndvi_url = get_ee_image_url(ndvi_median, BasemapConfig.NDVI_VIS_PARAMS)
                self.basemap_tiles['NDVI'] = ndvi_url

                # Add NDWI basemap
                ndwi_median = get_s2_ndwi_median(
                    self.ee_boundary, self.config.start_date, self.config.end_date)
                ndwi_url = get_ee_image_url(ndwi_median, BasemapConfig.NDWI_VIS_PARAMS)
                self.basemap_tiles['NDWI'] = ndwi_url
                
                print("✅ Earth Engine basemaps added successfully!")
                
            except Exception as e:
                print(f"⚠️  Failed to create Earth Engine basemaps: {e}")
                print("⚠️  Continuing with basic basemaps only")
        else:
            if not EE_AVAILABLE:
                print("⚠️  Earth Engine not available - NDVI/NDWI basemaps skipped")


    def _build_map(self, center_y, center_x):
        """Build and return the map widget."""
        map_widget = Map(
            basemap=self.basemap_layer,
            center=(center_y, center_x), 
            zoom=UIConstants.DEFAULT_ZOOM, 
            layout=Layout(flex='1 1 auto', height='100%'),
            scroll_wheel_zoom=True
        )
        return map_widget


    def _build_side_panel(self):
        """Build the collapsible side panel with accordion sections."""
        self.search_btn = Button(
            description='Search',
            layout=Layout(width='100%', height=UIConstants.BUTTON_HEIGHT),
            button_style='success',  # Green to highlight importance
            tooltip='Find points similar to your positive labels'
        )
        
        self.neighbors_slider = IntSlider(
            value=UIConstants.DEFAULT_NEIGHBORS,
            min=UIConstants.MIN_NEIGHBORS,
            max=UIConstants.MAX_NEIGHBORS,
            step=UIConstants.NEIGHBORS_STEP,
            description='',  # No description
            readout=True,
            layout=Layout(width='100%')
        )
        
        self.reset_btn = Button(
            description='🗑️ Reset',
            layout=Layout(width='100%', height=UIConstants.RESET_BUTTON_HEIGHT),
            button_style='',  # Default grey style
            tooltip='Clear all labels and search results'
        )
        
        search_section = VBox([
            self.search_btn,
            self.neighbors_slider,
            self.reset_btn
        ], layout=Layout(padding='5px', margin='0 0 10px 0'))
        
        # --- Labeling section ---
        self.label_toggle = ToggleButtons(
            options=[('Positive', 'Positive'), ('Negative', 'Negative'), ('Erase', 'Erase')],
            value='Positive',
            layout=Layout(width='100%')
        )
        
        # Add selection mode toggle
        self.selection_mode = ToggleButtons(
            options=[('Point', 'point'), ('Polygon', 'polygon')],
            value='point',
            layout=Layout(width='100%')
        )
        
        # Apply colors to toggle buttons
        self._update_toggle_button_styles()
        
        # --- Basemap Selection ---
        self.basemap_buttons = {}
        basemap_section_widgets = []
        
        # Use instance basemap_tiles which includes EE basemaps (NDVI/NDWI)
        basemap_tiles_to_use = getattr(self, 'basemap_tiles', BasemapConfig.BASEMAP_TILES)
        
        for basemap_name in basemap_tiles_to_use.keys():
            btn = Button(
                description=basemap_name.replace('_', ' '),
                layout=Layout(width='100%', margin='1px'),
                button_style=''
            )
            btn.basemap_name = basemap_name  # Store basemap name for reference
            self.basemap_buttons[basemap_name] = btn
            basemap_section_widgets.append(btn)
        
        # Highlight current basemap
        self._update_basemap_button_styles()
        
        # --- Export section ---
        self.save_btn = Button(description='💾 Save Dataset', layout=Layout(width='100%'))
        
        # --- Load Dataset section ---
        self.load_btn = Button(description='📂 Load Dataset', layout=Layout(width='100%'))
        self.file_upload = FileUpload(
            accept='.geojson,.parquet',
            multiple=False,
            layout=Layout(width='100%', display='none')  # Initially hidden
        )
        
        # --- External Tools section ---
        self.google_maps_btn = Button(
            description='🌍 Google Maps ↗',
            layout=Layout(width='100%'),
            button_style=''
        )
        
        # Build accordion
        accordion = Accordion(children=[
            VBox([
                Label('Label Type:'),
                self.label_toggle,
                Label('Selection Mode:', layout=Layout(margin='10px 0 0 0')),
                self.selection_mode
            ], layout=Layout(padding='5px')),
            VBox(basemap_section_widgets, layout=Layout(padding='5px')),
            VBox([self.save_btn, self.load_btn, self.file_upload, self.google_maps_btn], layout=Layout(padding='5px'))
        ])
        
        # Set titles
        for i, title in enumerate(['Label Mode', 'Basemaps', 'Export & Tools']):
            accordion.set_title(i, title)
        
        # Open label mode by default
        accordion.selected_index = 0
        
        # Add collapse/expand functionality
        self.panel_collapsed = False
        self.collapse_btn = Button(
            description='◀',
            layout=Layout(width=UIConstants.COLLAPSE_BUTTON_SIZE, height=UIConstants.COLLAPSE_BUTTON_SIZE),
            tooltip='Collapse/Expand Panel'
        )
        
        # Main panel with collapse button
        panel_header = HBox([
            Label('Controls', layout=Layout(flex='1')),
            self.collapse_btn
        ], layout=Layout(width='100%', justify_content='space-between', padding='2px'))
        
        # Create accordion container that will be hidden/shown
        self.accordion_container = VBox([accordion], layout=Layout(width='100%'))
        
        # Panel content includes search (always visible) and accordion (collapsible)
        panel_content = VBox([
            panel_header,
            search_section,  # Always visible
            self.accordion_container  # This will be hidden/shown
        ], layout=Layout(width=UIConstants.PANEL_WIDTH, padding='5px'))  # Narrower width
        
        # Return panel and widget references
        ui_widgets = {
            'search_btn': self.search_btn,
            'reset_btn': self.reset_btn,
            'label_toggle': self.label_toggle,
            'selection_mode': self.selection_mode,
            'neighbors_slider': self.neighbors_slider,
            'basemap_buttons': self.basemap_buttons,
            'save_btn': self.save_btn,
            'load_btn': self.load_btn,
            'file_upload': self.file_upload,
            'google_maps_btn': self.google_maps_btn,
            'collapse_btn': self.collapse_btn
        }
        
        return panel_content, ui_widgets


    def _update_toggle_button_styles(self):
        """Update toggle button colors based on selection."""
        style = """
        <style>
        .widget-toggle-buttons button:nth-child(1).mod-active {
            background-color: %s !important;
            color: white !important;
        }
        .widget-toggle-buttons button:nth-child(2).mod-active {
            background-color: %s !important;
            color: white !important;
        }
        .widget-toggle-buttons button:nth-child(3).mod-active {
            background-color: %s !important;
            color: white !important;
        }
        </style>
        """ % (UIConstants.POS_COLOR, UIConstants.NEG_COLOR, UIConstants.NEUTRAL_COLOR)
        display(HTML(style))


    def _add_map_layers(self):
        """Add all necessary layers to the map."""
        # Region boundary
        with open(self.config.boundary_path) as f:
            region_layer = ipyl.GeoJSON(
                    name="region",
                    data=json.load(f),
                    style=LayerStyles.get_region_style()
                )
        self.map.add_layer(region_layer)

        # Positive layer
        self.pos_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style=LayerStyles.get_point_style(UIConstants.POS_COLOR)
        )
        self.map.add_layer(self.pos_layer)

        # Negative layer
        self.neg_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style=LayerStyles.get_point_style(UIConstants.NEG_COLOR)
        )
        self.map.add_layer(self.neg_layer)

        # Erase layer
        self.erase_layer = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style=LayerStyles.get_erase_style()
        )
        self.map.add_layer(self.erase_layer)
        
        # Points layer for search results
        self.points = ipyl.GeoJSON(
            data=json.loads(gpd.GeoDataFrame(columns=['geometry']).to_json()),
            point_style=LayerStyles.get_search_style(),
            hover_style=LayerStyles.get_search_hover_style()
        )
        self.map.add_layer(self.points)
        

    def _setup_draw_control(self):
        """Set up the draw control for lasso selection."""
        self.draw_control = DrawControl(
            polygon=LayerStyles.get_draw_options(),
            polyline={},
            circle={},
            rectangle={},
            marker={},
            circlemarker={},
        )
        self.draw_control.on_draw(self.handle_draw)
        self.map.add_control(self.draw_control)
        self.draw_control.clear()

        # Track polygon drawing state
        self.polygon_drawing = False


    def _wire_events(self):
        """Wire all event handlers."""
        # Search button (main functionality)
        self.search_btn.on_click(self.search_click)
        
        # Reset button
        self.reset_btn.on_click(self.reset_all)
        
        # Label toggle
        self.label_toggle.observe(self._on_label_change, 'value')
        
        # Selection mode toggle
        self.selection_mode.observe(self._on_selection_mode_change, 'value')
        
        # # Neighbors slider
        # self.neighbors_slider.observe(self._on_neighbors_change, 'value')
        
        # Basemap buttons
        for basemap_name, btn in self.basemap_buttons.items():
            btn.on_click(lambda b, name=basemap_name: self._on_basemap_select(name))
        
        # Collapse button
        self.collapse_btn.on_click(self._on_toggle_collapse)
        
        # Export and external tools
        self.save_btn.on_click(self.save_dataset)
        self.load_btn.on_click(self._on_load_click)
        self.file_upload.observe(self._on_file_upload, names=['value'])
        self.google_maps_btn.on_click(self._on_google_maps_click)
        
        # Map interactions
        self.map.on_interaction(self._on_map_interaction)


    def _on_label_change(self, change):
        """Handle label toggle change."""
        self.current_label = change['new']
        if self.current_label == 'Positive':
            self.select_val = UIConstants.POSITIVE_LABEL
        elif self.current_label == 'Negative':
            self.select_val = UIConstants.NEGATIVE_LABEL
        else:  # Erase
            self.select_val = UIConstants.ERASE_LABEL
        self._update_status()


    def _on_google_maps_click(self, b):
        """Open current map center in Google Maps."""
        center = self.map.center
        url = f"https://www.google.com/maps/@{center[0]},{center[1]},15z"
        webbrowser.open(url, new=2)

    def _on_load_click(self, b):
        """Handle load dataset button click."""
        # Toggle file upload widget visibility
        if self.file_upload.layout.display == 'none':
            self.file_upload.layout.display = 'flex'
            self.load_btn.description = '📂 Cancel Load'
        else:
            self.file_upload.layout.display = 'none'
            self.load_btn.description = '📂 Load Dataset'
            # Clear any uploaded files
            self.file_upload.value = ()

    def _on_file_upload(self, change):
        """Handle file upload."""
        if not change['new']:
            return
        
        # Get the uploaded file - change['new'] is a tuple of uploaded files
        uploaded_files = change['new']
        if not uploaded_files:
            return
            
        # Get the first uploaded file
        uploaded_file = uploaded_files[0]
        filename = uploaded_file['name']
        content = uploaded_file['content']
        
        try:
            self.load_dataset_from_content(content, filename)
            # Hide the upload widget and reset button text
            self.file_upload.layout.display = 'none'
            self.load_btn.description = '📂 Load Dataset'
            # Clear the upload widget
            self.file_upload.value = ()
        except Exception as e:
            print(f"❌ Error loading file: {str(e)}")
            # Still hide the widget on error
            self.file_upload.layout.display = 'none'
            self.load_btn.description = '📂 Load Dataset'


    def _on_basemap_select(self, basemap_name):
        """Handle basemap selection."""
        self.current_basemap = basemap_name
        # Use instance basemap_tiles which includes EE basemaps
        if hasattr(self, 'basemap_tiles'):
            self.basemap_layer.url = self.basemap_tiles[basemap_name]
        else:
            self.basemap_layer.url = BasemapConfig.BASEMAP_TILES[basemap_name]
        self._update_basemap_button_styles()


    def _on_toggle_collapse(self, b):
        """Toggle panel collapse/expand."""
        if self.panel_collapsed:
            # Expand
            self.accordion_container.layout.display = 'flex'
            self.collapse_btn.description = '◀'
            self.panel_collapsed = False
        else:
            # Collapse
            self.accordion_container.layout.display = 'none'
            self.collapse_btn.description = '▶'
            self.panel_collapsed = True


    def _on_map_interaction(self, **kwargs):
        """Handle all map interactions."""
        lat, lon = kwargs.get('coordinates', (0, 0))
        
        # Update status
        self._update_status(lat, lon)
        
        # Handle shift-click for polygon drawing hint
        if kwargs.get('type') == 'mousemove' and kwargs.get('modifiers', {}).get('shiftKey', False):
            self.status_bar.value += " | <b>Hold Shift + Draw to select multiple points</b>"
        
        # Handle ctrl-click for Google Maps
        if kwargs.get('type') == 'click' and kwargs.get('modifiers', {}).get('ctrlKey', False):
            url = f"https://www.google.com/maps/@{lat},{lon},18z"
            webbrowser.open(url, new=2)
            return
        
        # Normal label point behavior
        self.label_point(**kwargs)


    def _on_selection_mode_change(self, change):
        """Handle selection mode change."""
        self.lasso_mode = (change['new'] == 'polygon')
        self._update_status()


    def handle_draw(self, target, action, geo_json):
        """Handle polygon drawing with automatic mode switching."""
        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            # Mark that we're processing a polygon
            self.polygon_drawing = False
            
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
                
                if self.select_val == UIConstants.POSITIVE_LABEL:
                    self.pos_ids.append(point_id)
                elif self.select_val == UIConstants.NEGATIVE_LABEL:
                    self.neg_ids.append(point_id)
            
            self.update_layers()
            self.update_query_vector()
            
            # Clear the polygon after processing
            self.draw_control.clear()
            self._update_status()
        
        elif action == 'drawstart':
            # Mark that we're starting to draw a polygon
            if self.lasso_mode:
                self.polygon_drawing = True
                self._update_status()
        
        elif action == 'deleted':
            # Reset polygon drawing state
            self.polygon_drawing = False
            self._update_status()


    def _update_status(self, lat=None, lon=None):
        """Update the status bar."""
        if lat is None or lon is None:
            center = self.map.center
            lat, lon = center[0], center[1]
        
        mode = "Polygon" if self.lasso_mode else "Point"
        label = self.current_label
        
        status_text = f"Lat: {lat:.4f} | Lon: {lon:.4f} | Mode: {mode} | Label: {label}"
        
        if self.lasso_mode:
            if self.polygon_drawing:
                status_text += " | <b>Drawing polygon...</b>"
        
        self.status_bar.value = f"""
            <div style='background: white; padding: 5px; border-radius: 5px; opacity: 0.8; font-size: 12px;'>
                {status_text}
            </div>
        """

    def reset_all(self, b):
        """Reset all labels, search results, and cached data."""
        print("🗑️ Resetting all labels and search results...")
        
        # Clear all label lists
        self.pos_ids = []
        self.neg_ids = []
        
        # Clear cached embeddings
        self.cached_embeddings = {}
        
        # Reset query vector
        self.query_vector = None
        
        # Clear detections
        self.detections_with_embeddings = None
        
        # Clear all map layers
        empty_geojson = {"type": "FeatureCollection", "features": []}
        self.pos_layer.data = empty_geojson
        self.neg_layer.data = empty_geojson
        self.erase_layer.data = empty_geojson
        self.points.data = empty_geojson
        
        print("✅ All data cleared!")

    def search_click(self, b):
        """Perform similarity search based on current query vector."""
        if self.query_vector is None:
            print("⚠️ No query vector available. Please add some positive labels first.")
            return
        
        n_neighbors = self.neighbors_slider.value
        
        # Convert query vector to the format needed for DuckDB
        query_vec = self.query_vector.tolist()
        
        sql = DatabaseConstants.SIMILARITY_SEARCH_QUERY
        
        print(f"🔍 Searching for {n_neighbors} similar points...")
        search_results = self.duckdb_connection.execute(sql, [query_vec, n_neighbors]).df()
        
        search_results_filtered = search_results[~search_results['id'].isin(self.pos_ids)]
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
        

    def label_point(self, **kwargs):
        """Assign a label and map layer to a clicked map point."""
        # Don't process clicks when in polygon mode or actively drawing
        if not self.execute_label_point or self.lasso_mode or self.polygon_drawing:
            return
        
        action = kwargs.get('type') 
        if action not in ['click']:
            return
                 
        lat, lon = kwargs.get('coordinates')
        
        clicked_point = Point(lon, lat)
        point_id = None
        embedding = None
        
        # First check if we have cached detections
        if self.detections_with_embeddings is not None and len(self.detections_with_embeddings) > 0:
            # Find nearest point in cached detections
            distances = self.detections_with_embeddings.geometry.distance(clicked_point)
            nearest_idx = distances.idxmin()
            
            # Use a threshold to ensure we're clicking on an actual point
            if distances[nearest_idx] < UIConstants.CLICK_THRESHOLD:  # Adjust threshold as needed
                nearest_detection = self.detections_with_embeddings.loc[nearest_idx]
                point_id = nearest_detection['id']
                embedding = nearest_detection['embedding']
        
        # If not found in cache, query the database
        if point_id is None:
            sql = DatabaseConstants.NEAREST_POINT_QUERY
            
            nearest_result = self.duckdb_connection.execute(sql, [lon, lat]).df()
            
            if nearest_result.empty:
                return
            
            point_id = str(nearest_result.iloc[0]['id'])  # Convert to string
            embedding = nearest_result.iloc[0]['embedding']
        
        # Cache the embedding for later use
        self.cached_embeddings[point_id] = np.array(embedding)
        
        # Update labels
        if point_id in self.pos_ids:
            self.pos_ids.remove(point_id)
        if point_id in self.neg_ids:
            self.neg_ids.remove(point_id)
                
        if self.select_val == UIConstants.POSITIVE_LABEL:
            self.pos_ids.append(point_id)
        elif self.select_val == UIConstants.NEGATIVE_LABEL:
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
            
            # Determine label (positive or negative)
            if point_id in self.pos_ids:
                label = UIConstants.POSITIVE_LABEL
            elif point_id in self.neg_ids:
                label = UIConstants.NEGATIVE_LABEL
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
                "positive_points": len([f for f in features if f['properties']['label'] == UIConstants.POSITIVE_LABEL]),
                "negative_points": len([f for f in features if f['properties']['label'] == UIConstants.NEGATIVE_LABEL]),
                "embedding_dimension": len(features[0]['properties']['embedding']) if features else 0
            }
        }
        
        # Save to file
        filename = f"labeled_dataset_{timestamp}.geojson"
        
        try:
            with open(filename, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            # Create summary
            pos_count = len([f for f in features if f['properties']['label'] == UIConstants.POSITIVE_LABEL])
            neg_count = len([f for f in features if f['properties']['label'] == UIConstants.NEGATIVE_LABEL])
            
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
                if label == UIConstants.POSITIVE_LABEL:
                    self.pos_ids.append(point_id)
                elif label == UIConstants.NEGATIVE_LABEL:
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

    def load_dataset_from_content(self, content, filename):
        """Load a dataset from uploaded file content."""
        print(f"📂 Loading dataset from {filename}...")
        
        try:
            # Convert content to bytes if it's a memoryview
            if isinstance(content, memoryview):
                content_bytes = content.tobytes()
            elif isinstance(content, bytes):
                content_bytes = content
            else:
                content_bytes = bytes(content)
            
            # Determine file type and parse accordingly
            if filename.lower().endswith('.geojson'):
                # Parse GeoJSON
                geojson_data = json.loads(content_bytes.decode('utf-8'))
                self._process_geojson_data(geojson_data, filename)
                
            elif filename.lower().endswith('.parquet'):
                # Parse GeoParquet using pandas/geopandas
                import io
                gdf = gpd.read_parquet(io.BytesIO(content_bytes))
                self._process_geoparquet_data(gdf, filename)
                
            else:
                raise ValueError(f"Unsupported file format. Please use .geojson or .parquet files.")
                
        except Exception as e:
            raise Exception(f"Error processing {filename}: {str(e)}")

    def _process_geojson_data(self, geojson_data, filename):
        """Process GeoJSON data and populate labels."""
        # Clear current labels
        self.pos_ids = []
        self.neg_ids = []
        self.cached_embeddings = {}
        
        # Process features
        for feature in geojson_data['features']:
            point_id = str(feature['properties']['id'])  # Ensure string type
            label = feature['properties']['label']
            embedding = np.array(feature['properties']['embedding'])
            
            # Cache the embedding
            self.cached_embeddings[point_id] = embedding
            
            # Add to appropriate list
            if label == UIConstants.POSITIVE_LABEL:
                self.pos_ids.append(point_id)
            elif label == UIConstants.NEGATIVE_LABEL:
                self.neg_ids.append(point_id)
        
        # Update visualization
        self.update_layers()
        self.update_query_vector()
        
        # Print summary
        metadata = geojson_data.get('metadata', {})
        print(f"✅ Dataset loaded successfully from {filename}!")
        print(f"📊 Summary:")
        print(f"   - Total points: {metadata.get('total_points', len(geojson_data['features']))}")
        print(f"   - Positive labels: {len(self.pos_ids)}")
        print(f"   - Negative labels: {len(self.neg_ids)}")
        print(f"   - Saved on: {metadata.get('timestamp', 'Unknown')}")

    def _process_geoparquet_data(self, gdf, filename):
        """Process GeoParquet data and populate labels."""
        # Clear current labels
        self.pos_ids = []
        self.neg_ids = []
        self.cached_embeddings = {}
        
        # Check required columns
        required_cols = ['id', 'label', 'embedding']
        for col in required_cols:
            if col not in gdf.columns:
                raise ValueError(f"Required column '{col}' not found in {filename}")
        
        # Process each row
        for _, row in gdf.iterrows():
            point_id = str(row['id'])  # Ensure string type
            label = row['label']
            
            # Handle embedding - could be stored as array or list
            if isinstance(row['embedding'], (list, np.ndarray)):
                embedding = np.array(row['embedding'])
            else:
                # Try to parse if it's stored as string
                embedding = np.array(json.loads(row['embedding']))
            
            # Cache the embedding
            self.cached_embeddings[point_id] = embedding
            
            # Add to appropriate list
            if label == UIConstants.POSITIVE_LABEL:
                self.pos_ids.append(point_id)
            elif label == UIConstants.NEGATIVE_LABEL:
                self.neg_ids.append(point_id)
        
        # Update visualization
        self.update_layers()
        self.update_query_vector()
        
        # Print summary
        print(f"✅ Dataset loaded successfully from {filename}!")
        print(f"📊 Summary:")
        print(f"   - Total points: {len(gdf)}")
        print(f"   - Positive labels: {len(self.pos_ids)}")
        print(f"   - Negative labels: {len(self.neg_ids)}")

    def _update_basemap_button_styles(self):
        """Update basemap button styles to highlight current selection."""
        for basemap_name, btn in self.basemap_buttons.items():
            if basemap_name == self.current_basemap:
                btn.button_style = 'info'  # Blue highlight for active
            else:
                btn.button_style = ''  # Default style

    def close(self):
        """Clean up resources."""
        if hasattr(self, '_owns_connection') and self._owns_connection:
            if hasattr(self, 'duckdb_connection') and self.duckdb_connection:
                self.duckdb_connection.close()
                print("🔌 DuckDB connection closed.")
