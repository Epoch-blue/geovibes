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
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

import yaml


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
    module_root = Path(__file__).resolve().parent
    src_dir = module_root / "src"

    import_paths: list[str] = [str(module_root)]
    if src_dir.exists():
        import_paths.append(str(src_dir))

    try:
        import site

        import_paths.extend(site.getsitepackages())
    except Exception:
        import sysconfig

        paths = sysconfig.get_paths()
        for key in ("purelib", "platlib"):
            path = paths.get(key)
            if path:
                import_paths.append(path)

    seen: set[str] = set()
    init_source = ["import sys\n"]
    for path in import_paths:
        if path not in seen:
            init_source.append(f"sys.path.insert(0, {repr(path)})\n")
            seen.add(path)

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
                "display_name": "GeoVibes (Voila)",
                "language": "python",
                "name": "geovibes-voila",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": sys.version.split(" ")[0],
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
    kernels_dir = Path(temp_dir) / "kernels"
    kernel_name = "geovibes-voila"
    kernel_dir = kernels_dir / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "GeoVibes (Voila)",
        "language": "python",
    }
    with open(kernel_dir / "kernel.json", "w") as kernel_file:
        json.dump(kernel_spec, kernel_file, indent=2)

    # Cleanup function
    def cleanup():
        shutil.rmtree(temp_dir, ignore_errors=True)

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

    voila_env = os.environ.copy()
    existing_jupyter_path = voila_env.get("JUPYTER_PATH")
    kernels_path = str(kernels_dir)
    if existing_jupyter_path:
        voila_env["JUPYTER_PATH"] = os.pathsep.join([kernels_path, existing_jupyter_path])
    else:
        voila_env["JUPYTER_PATH"] = kernels_path

    process = subprocess.Popen(voila_cmd, env=voila_env)

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


def ensure_uv_runtime():
    """Re-exec the script with `uv run` if not already active."""
    if os.environ.get("UV_RUN_RECURSION_DEPTH"):
        return

    uv_executable = shutil.which("uv")
    if not uv_executable:
        return

    script_path = Path(__file__).resolve()
    if not script_path.exists():
        return

    args = ["uv", "run", str(script_path), *sys.argv[1:]]
    os.execvp("uv", args)


def main():
    """Main entry point for the GeoVibes web application."""
    ensure_uv_runtime()
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
