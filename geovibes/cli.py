#!/usr/bin/env python3
"""
Command line interface for GeoVibes web application.
"""

import sys
from pathlib import Path


def main():
    """Main entry point for the geovibes-webapp command."""
    # Add the project root to Python path so we can import run_geovibes_webapp
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Import and run the existing webapp script
    try:
        from run_geovibes_webapp import main as webapp_main

        webapp_main()
    except ImportError:
        print("Error: Could not import run_geovibes_webapp module.", file=sys.stderr)
        print(
            "Make sure you're running from the project root directory.", file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
