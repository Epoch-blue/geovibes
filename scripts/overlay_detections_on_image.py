"""Overlay GeoJSON detections on a georeferenced map image.

This script takes a map image and overlays detection polygons by
mapping geographic coordinates to pixel coordinates.
"""

import argparse
from pathlib import Path

import geopandas as gpd
from PIL import Image, ImageDraw, ImageFont


def geo_to_pixel(
    lon: float,
    lat: float,
    geo_bounds: tuple,
    pixel_bounds: tuple,
) -> tuple:
    """Convert geographic coordinates to pixel coordinates.

    Args:
        lon: Longitude
        lat: Latitude
        geo_bounds: (min_lon, min_lat, max_lon, max_lat)
        pixel_bounds: (left, top, right, bottom) in pixels

    Returns:
        (x, y) pixel coordinates
    """
    min_lon, min_lat, max_lon, max_lat = geo_bounds
    px_left, px_top, px_right, px_bottom = pixel_bounds

    # Normalize to 0-1
    x_norm = (lon - min_lon) / (max_lon - min_lon)
    y_norm = (lat - min_lat) / (max_lat - min_lat)

    # Convert to pixels (y is inverted in images)
    px_x = px_left + x_norm * (px_right - px_left)
    px_y = px_bottom - y_norm * (px_bottom - px_top)

    return px_x, px_y


def polygon_to_pixels(geom, geo_bounds, pixel_bounds) -> list:
    """Convert a polygon geometry to pixel coordinates."""
    if geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
        return [geo_to_pixel(lon, lat, geo_bounds, pixel_bounds) for lon, lat in coords]
    elif geom.geom_type == "MultiPolygon":
        all_coords = []
        for poly in geom.geoms:
            coords = list(poly.exterior.coords)
            all_coords.append(
                [
                    geo_to_pixel(lon, lat, geo_bounds, pixel_bounds)
                    for lon, lat in coords
                ]
            )
        return all_coords
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Overlay GeoJSON detections on a map image"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the map image",
    )
    parser.add_argument(
        "--geojson",
        required=True,
        help="Path to detection GeoJSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label text to add to the image",
    )
    parser.add_argument(
        "--geo-bounds",
        nargs=4,
        type=float,
        default=[-88.473227, 30.144425, -84.888246, 35.008028],
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Geographic bounds of the map area",
    )
    parser.add_argument(
        "--pixel-bounds",
        nargs=4,
        type=int,
        default=None,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Pixel bounds of the map area (auto-detect if not provided)",
    )
    parser.add_argument(
        "--color",
        default="red",
        help="Color for detection outlines (default: red)",
    )
    parser.add_argument(
        "--fill-alpha",
        type=int,
        default=80,
        help="Fill transparency 0-255 (default: 80)",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Outline width in pixels (default: 2)",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=0,
        help="Buffer geometries by this many meters before plotting (default: 0)",
    )

    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    img = Image.open(args.image).convert("RGBA")
    img_width, img_height = img.size
    print(f"  Image size: {img_width} x {img_height}")

    # Pixel bounds for the Alabama soil map
    # These were measured manually from the image
    # The map area (excluding title, legend, scale bar)
    if args.pixel_bounds:
        pixel_bounds = tuple(args.pixel_bounds)
    else:
        # Default bounds for the USDA 1993 Alabama Soil Map
        # Measured from the image: map starts around x=47, ends around x=598
        # y starts around 38, ends around 789
        pixel_bounds = (47, 38, 598, 789)
        print(f"  Using default pixel bounds for Alabama soil map: {pixel_bounds}")

    geo_bounds = tuple(args.geo_bounds)
    print(f"  Geographic bounds: {geo_bounds}")

    print(f"\nLoading detections: {args.geojson}")
    detections = gpd.read_file(args.geojson)
    print(f"  Loaded {len(detections)} features")

    # Buffer geometries if requested
    if args.buffer_m > 0:
        print(f"  Buffering by {args.buffer_m}m...")
        # Project to UTM for accurate buffering
        centroid = detections.union_all().centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        utm_crs = (
            f"EPSG:326{utm_zone:02d}" if centroid.y >= 0 else f"EPSG:327{utm_zone:02d}"
        )
        detections = detections.to_crs(utm_crs)
        detections["geometry"] = detections.geometry.buffer(args.buffer_m)
        detections = detections.to_crs("EPSG:4326")

    # Create overlay layer with transparency
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Parse color
    from PIL import ImageColor

    try:
        rgb = ImageColor.getrgb(args.color)
    except ValueError:
        rgb = (255, 0, 0)  # Default to red

    fill_color = (*rgb, args.fill_alpha)
    outline_color = (*rgb, 255)

    print(f"\nDrawing {len(detections)} detection polygons...")
    for idx, row in detections.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        if geom.geom_type == "Polygon":
            pixel_coords = polygon_to_pixels(geom, geo_bounds, pixel_bounds)
            if pixel_coords:
                draw.polygon(
                    pixel_coords,
                    fill=fill_color,
                    outline=outline_color,
                    width=args.line_width,
                )
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = list(poly.exterior.coords)
                pixel_coords = [
                    geo_to_pixel(lon, lat, geo_bounds, pixel_bounds)
                    for lon, lat in coords
                ]
                if pixel_coords:
                    draw.polygon(
                        pixel_coords,
                        fill=fill_color,
                        outline=outline_color,
                        width=args.line_width,
                    )
        elif geom.geom_type == "Point":
            px, py = geo_to_pixel(geom.x, geom.y, geo_bounds, pixel_bounds)
            r = 4
            draw.ellipse(
                [px - r, py - r, px + r, py + r], fill=fill_color, outline=outline_color
            )

    # Composite overlay onto image
    result = Image.alpha_composite(img, overlay)

    # Add legend entry for detections (below existing soil legend)
    if args.label:
        draw_result = ImageDraw.Draw(result)
        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except OSError:
            font = ImageFont.load_default()

        # Position below the existing soil legend entries
        # Matching the style of existing legend: colored box on left, text on right
        # Legend entries: Limestone Valley, Appalachian Plateau, Piedmont Plateau,
        # Blackland Prairie, Coastal Plain, Major Flood Plain, Coastal Marsh and Beach
        legend_x = 695  # X position for legend box (aligned with existing boxes)
        legend_y = 345  # Y position below COASTAL MARSH AND BEACH (the last entry)
        box_width = 30
        box_height = 16
        text_offset = 36

        # Draw legend box with detection color (matching existing box style)
        draw_result.rectangle(
            [legend_x, legend_y, legend_x + box_width, legend_y + box_height],
            fill=(*rgb, 220),
            outline=(0, 0, 0),
            width=1,
        )

        # Draw legend text
        draw_result.text(
            (legend_x + text_offset, legend_y + 2),
            args.label.upper(),
            fill=(0, 0, 0),
            font=font,
        )

    # Save result
    output_path = Path(args.output)
    if output_path.suffix.lower() in [".jpg", ".jpeg"]:
        result = result.convert("RGB")
    result.save(args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
