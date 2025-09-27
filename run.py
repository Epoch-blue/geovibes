#!/usr/bin/env python3
"""
GeoVibes Web Application Runner

This script creates a standalone web application from the GeoVibes interface
that can be accessed via a web browser instead of Jupyter notebook.

Usage:
    python run.py --config config.yaml
    python run.py --enable-ee
    python run.py --help

The script will start a web server and open the GeoVibes interface in your default browser.
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path
import yaml
import tempfile
import atexit
import subprocess
import json


def parse_arguments():
    """Parse command line arguments for GeoVibes configuration."""
    parser = argparse.ArgumentParser(
        description="Run GeoVibes as a standalone web application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use YAML config file
  python run.py --config config.yaml

  # Override defaults for basemap dates
  python run.py --start-date 2024-06-01 --end-date 2024-12-31
        """,
    )

    # Configuration file options
    parser.add_argument(
        "--config", type=str, help="Path to configuration file (YAML or JSON format)"
    )

    # Date options
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date for Earth Engine basemaps (YYYY-MM-DD format, default: 2024-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-01-01",
        help="End date for Earth Engine basemaps (YYYY-MM-DD format, default: 2025-01-01)",
    )

    # Google Cloud options
    parser.add_argument(
        "--gcp-project",
        type=str,
        help="Google Cloud Project ID for Earth Engine authentication",
    )
    parser.add_argument(
        "--enable-ee",
        action="store_true",
        default=False,
        help="Opt in to Earth Engine basemaps (requires prior earthengine authenticate)",
    )
    parser.add_argument(
        "--disable-ee",
        action="store_true",
        default=False,
        help="Force Earth Engine basemaps off even if config enables them",
    )

    # Web server options
    parser.add_argument(
        "--port",
        type=int,
        default=8866,
        help="Port to run the web server on (default: 8866)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the web server to (default: localhost)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not automatically open browser"
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def load_config_file(config_path):
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file type: {config_path.suffix}")


def merge_config(file_config, args):
    """Merge file configuration with command line arguments."""
    # Start with file config
    config = file_config.copy() if file_config else {}

    # Override with command line arguments (only if they were explicitly provided)
    arg_mappings = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "gcp_project": args.gcp_project,
    }

    for key, value in arg_mappings.items():
        if value is not None:
            config[key] = value

    if args.enable_ee:
        config["enable_ee"] = True
    if args.disable_ee:
        config["enable_ee"] = False

    return config


def sanitize_config(config: dict) -> dict:
    """Remove empty entries and deprecated CLI keys before notebook init."""
    sanitized = {k: v for k, v in config.items() if v is not None}

    # Remove deprecated path-based configuration keys
    sanitized.pop("duckdb_directory", None)
    sanitized.pop("duckdb_path", None)
    sanitized.pop("boundary_path", None)

    return sanitized


def create_notebook_content(config, verbose=False, disable_ee=False):
    """Create a temporary notebook that initializes GeoVibes with the given config."""
    project_root = str(Path(__file__).resolve().parent)
    src_dir = Path(project_root, "src")

    init_source = [
        "import sys\n",
        f"sys.path.insert(0, {repr(project_root)})\n",
    ]

    if src_dir.exists():
        init_source.append(f"sys.path.insert(0, {repr(str(src_dir))})\n")

    init_source.extend(
        [
            "from geovibes.ui import GeoVibes\n",
            f"config = {repr(config)}\n",
            f"verbose = {verbose}\n",
            f"disable_ee = {disable_ee}\n",
            "vibes = GeoVibes(\n",
            "    config=config,\n",
            "    verbose=verbose,\n",
            "    disable_ee=disable_ee\n",
            ")\n",
        ]
    )

    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": init_source,
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    return notebook_content


def run_with_voila(config, args):
    """Run the application using Voila."""

    # Create temporary notebook
    notebook_content = create_notebook_content(
        config,
        args.verbose,
        args.disable_ee,
    )

    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_notebook = os.path.join(temp_dir, "geovibes_app.ipynb")

    # Cleanup function
    def cleanup():
        try:
            os.remove(temp_notebook)
            os.rmdir(temp_dir)
        except:
            pass

    atexit.register(cleanup)

    # Write notebook content
    with open(temp_notebook, "w") as f:
        json.dump(notebook_content, f, indent=2)

    print(f"🚀 Starting GeoVibes web application on http://{args.host}:{args.port}")
    print(f"📊 Configuration: {config}")

    if not args.no_browser:
        # Give the server a moment to start before opening browser
        import threading
        import time

        def open_browser():
            time.sleep(3)
            webbrowser.open(f"http://{args.host}:{args.port}")

        threading.Thread(target=open_browser, daemon=True).start()

    print("🔧 Starting Voila server...")

    # Build Voila command with error suppression
    voila_cmd = [
        sys.executable,
        "-m",
        "voila",
        temp_notebook,
        "--port",
        str(args.port),
        "--no-browser",
        f"--Voila.server_host={args.host}",
        "--VoilaConfiguration.show_tracebacks=True",
    ]

    process = subprocess.Popen(voila_cmd)

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down GeoVibes web application...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        cleanup()


def main():
    """Main entry point for the GeoVibes web application."""
    args = parse_arguments()

    if args.enable_ee and args.disable_ee:
        print("❌ --enable-ee and --disable-ee cannot be used together")
        sys.exit(1)

    # Load configuration
    file_config = {}
    if args.config:
        try:
            file_config = load_config_file(args.config)
            if args.verbose:
                print(f"✅ Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            sys.exit(1)

    # Merge file config with command line arguments
    config = merge_config(file_config, args)
    config = sanitize_config(config)

    if args.verbose:
        print("🔧 Final configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")

    run_with_voila(config, args)


if __name__ == "__main__":
    main()
