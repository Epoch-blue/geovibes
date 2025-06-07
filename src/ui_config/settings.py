"""
Configuration management for GeoVibes.
"""

import json
import os
from dataclasses import dataclass


@dataclass
class GeoVibesConfig:
    """Configuration for GeoVibes."""
    
    duckdb_path: str
    boundary_path: str
    start_date: str
    end_date: str
    
    @classmethod
    def from_file(cls, config_path: str) -> 'GeoVibesConfig':
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            GeoVibesConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required fields are missing
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Validate required fields
        required_fields = ['duckdb_path', 'boundary_path', 'start_date', 'end_date']
        missing_fields = [field for field in required_fields if field not in config_data]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        return cls(**{k: v for k, v in config_data.items() if k in required_fields})
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GeoVibesConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration
            
        Returns:
            GeoVibesConfig instance
        """
        return cls(
            duckdb_path=config_dict['duckdb_path'],
            boundary_path=config_dict['boundary_path'],
            start_date=config_dict['start_date'],
            end_date=config_dict['end_date']
        )
    
    def validate(self) -> None:
        """Validate the configuration.
        
        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If configuration values are invalid
        """
        # Check that required files exist
        if not os.path.exists(self.boundary_path):
            raise FileNotFoundError(f"Boundary file not found: {self.boundary_path}")
        
        if not os.path.exists(self.duckdb_path):
            raise FileNotFoundError(f"DuckDB file not found: {self.duckdb_path}")
        
        # Validate date format (basic check)
        import datetime
        try:
            datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
            datetime.datetime.strptime(self.end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
        
        # Check that start_date is before end_date
        start = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(self.end_date, '%Y-%m-%d')
        
        if start >= end:
            raise ValueError("start_date must be before end_date")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'duckdb_path': self.duckdb_path,
            'boundary_path': self.boundary_path,
            'start_date': self.start_date,
            'end_date': self.end_date
        } 