#!/usr/bin/env python3
"""
GeoVibes Web Application Runner

This script creates a standalone web application from the GeoVibes interface
that can be accessed via a web browser instead of Jupyter notebook.

Usage:
    python run_geovibes_webapp.py --config config.yaml
    python run_geovibes_webapp.py --duckdb-directory ./local_databases --boundary geometries/alabama.geojson
    python run_geovibes_webapp.py --help

The script will start a web server and open the GeoVibes interface in your default browser.
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path
import yaml
import json
import tempfile
import atexit
import subprocess


def parse_arguments():
    """Parse command line arguments for GeoVibes configuration."""
    parser = argparse.ArgumentParser(
        description="Run GeoVibes as a standalone web application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use YAML config file
  python run_geovibes_webapp.py --config config.yaml
  
  # Use individual parameters
  python run_geovibes_webapp.py --duckdb-directory ./local_databases --boundary geometries/alabama.geojson
  
  # Use JSON config file (legacy)
  python run_geovibes_webapp.py --config config/resnet_alabama_config.json
  
  # Override config with individual parameters
  python run_geovibes_webapp.py --config config.yaml --verbose --start-date 2024-06-01
        """,
    )

    # Configuration file options
    parser.add_argument(
        "--config", type=str, help="Path to configuration file (YAML or JSON format)"
    )

    # Database options
    parser.add_argument("--duckdb-path", type=str, help="Path to DuckDB database file")
    parser.add_argument(
        "--duckdb-directory",
        type=str,
        help="Directory containing DuckDB database files",
    )

    # Geographic options
    parser.add_argument(
        "--boundary",
        "--boundary-path",
        dest="boundary_path",
        type=str,
        help="Path to boundary GeoJSON file",
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
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            return json.load(f)
        else:
            # Try to detect format by content
            content = f.read()
            f.seek(0)
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError:
                f.seek(0)
                return json.load(f)


def merge_config(file_config, args):
    """Merge file configuration with command line arguments."""
    # Start with file config
    config = file_config.copy() if file_config else {}

    # Override with command line arguments (only if they were explicitly provided)
    arg_mappings = {
        "duckdb_path": args.duckdb_path,
        "duckdb_directory": args.duckdb_directory,
        "boundary_path": args.boundary_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "gcp_project": args.gcp_project,
    }

    for key, value in arg_mappings.items():
        if value is not None:
            config[key] = value

    return config


def create_notebook_content(config, verbose=False):
    """Create a temporary notebook that initializes GeoVibes with the given config."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🌍 GeoVibes: Explore Earth Observation Embeddings\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Auto-generated GeoVibes initialization\n",
                    "import sys\n",
                    "import os\n",
                    "\n",
                    "# Add src directory to path\n",
                    "sys.path.insert(0, os.path.join(os.getcwd(), 'src'))\n",
                    "\n",
                    "from geovibes.ui import GeoVibes\n",
                    "\n",
                    "# Initialize GeoVibes with configuration\n",
                    f"config = {repr(config)}\n",
                    f"verbose = {verbose}\n",
                    "\n",
                    "vibes = GeoVibes(\n",
                    "    duckdb_path=config.get('duckdb_path'),\n",
                    "    duckdb_directory=config.get('duckdb_directory'),\n",
                    "    boundary_path=config.get('boundary_path'),\n",
                    "    start_date=config.get('start_date', '2024-01-01'),\n",
                    "    end_date=config.get('end_date', '2025-01-01'),\n",
                    "    gcp_project=config.get('gcp_project'),\n",
                    "    verbose=verbose\n",
                    ")",
                ],
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
    notebook_content = create_notebook_content(config, args.verbose)

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

    # Try subprocess approach first (most reliable)
    try:
        print("🔧 Starting Voila server...")

        # Build Voila command with error suppression
        voila_cmd = [
            sys.executable,
            "-m",
            "voila",
            temp_notebook,
            "--port",
            str(args.port),
            "--ip",
            args.host,
            "--no-browser",
        ]

        # Run Voila as subprocess with error suppression
        with open(os.devnull, "w") as devnull:
            # Redirect stderr to capture and filter errors
            process = subprocess.Popen(
                voila_cmd,
                stderr=subprocess.PIPE if not args.verbose else None,
                universal_newlines=True,
            )

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

    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        if args.verbose:
            print(f"⚠️  Subprocess Voila failed: {e}")
        print("🔄 Trying Python API...")

        # Fall back to Python API approach
        try:
            from voila.app import Voila

            app = Voila()

            # Set configuration with error handling
            if hasattr(app, "port"):
                app.port = args.port
            if hasattr(app, "ip"):
                app.ip = args.host
            if hasattr(app, "notebook_path"):
                app.notebook_path = temp_notebook
            elif hasattr(app, "notebook"):
                app.notebook = temp_notebook

            app.open_browser = False

            # Configure logging level
            if hasattr(app, "log_level"):
                app.log_level = "WARN"

            if hasattr(app, "enable_nbextensions"):
                app.enable_nbextensions = True

            try:
                app.start()
            except KeyboardInterrupt:
                print("\n🛑 Shutting down GeoVibes web application...")
                cleanup()
            except Exception as api_error:
                if args.verbose:
                    print(f"⚠️  Voila Python API failed: {api_error}")
                print("🔄 Falling back to standalone mode...")
                raise

        except Exception as fallback_error:
            if args.verbose:
                print(f"❌ All Voila methods failed: {fallback_error}")
            raise


def main():
    """Main entry point for the GeoVibes web application."""
    args = parse_arguments()

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

    # Validate configuration
    if not config.get("duckdb_path") and not config.get("duckdb_directory"):
        print("❌ Error: Either --duckdb-path or --duckdb-directory must be provided")
        sys.exit(1)

    if args.verbose:
        print("🔧 Final configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")

    run_with_voila(config, args)


if __name__ == "__main__":
    main()
