"""Configuration management for GeoVibes."""

import os
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class GeoVibesConfig:
    """Minimal configuration for GeoVibes."""

    start_date: str = "2024-01-01"
    end_date: str = "2025-01-01"
    gcp_project: Optional[str] = None

    @classmethod
    def from_file(cls, config_path: str) -> "GeoVibesConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GeoVibesConfig":
        """Create configuration from dictionary."""
        return cls(
            start_date=config_dict.get("start_date", "2024-01-01"),
            end_date=config_dict.get("end_date", "2025-01-01"),
            gcp_project=config_dict.get("gcp_project"),
        )

    def validate(self) -> None:
        """Validate the configuration values."""
        import datetime

        try:
            start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {exc}")

        if start >= end:
            raise ValueError("start_date must be before end_date")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        config_dict = {
            "start_date": self.start_date,
            "end_date": self.end_date,
        }
        if self.gcp_project:
            config_dict["gcp_project"] = self.gcp_project
        return config_dict
