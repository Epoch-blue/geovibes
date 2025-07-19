# Makefile for GeoVibes

# Use the conda environment python
PYTHON := $(shell which python || echo python3)

.PHONY: setup run webapp

# Setup the conda environment
setup:
	@echo "Creating conda environment..."
	@mamba create -n geovibes python=3.11 -y
	@echo "To activate the environment, run: mamba activate geovibes"
	@echo "Installing dependencies..."
	@mamba install -c conda-forge --file ./requirements.txt -y
	@echo "Setup complete!"

# Run the NiceGUI application
run:
	@echo "Launching GeoVibes..."
	@$(PYTHON) -m geovibes.nicegui_app 

# Run the GeoVibes web application
webapp:
	@echo "Launching GeoVibes web application..."
	@$(PYTHON) run.py --config config.yaml

# Run the GeoVibes web application with custom config
webapp-config:
	@echo "Usage: make webapp-config CONFIG=path/to/config.yaml"
	@$(PYTHON) run.py --config $(CONFIG)