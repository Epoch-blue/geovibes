# Makefile for GeoVibes

# Use the conda environment python
PYTHON := $(shell which python || echo python3)

.PHONY: setup run

# Setup the conda environment
setup:
	@echo "Creating conda environment..."
	@mamba create -n geovibes python=3.11 -y
	@echo "To activate the environment, run: mamba activate geovibes"
	@echo "Installing dependencies..."
	@pip install -e ".[all]"
	@echo "Setup complete!"

# Run the NiceGUI application
run:
	@echo "Launching GeoVibes..."
	@$(PYTHON) -m geovibes.nicegui_app 
