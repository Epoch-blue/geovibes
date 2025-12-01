"""
GeoVibes: check out your satellite foundation model's vibes
"""

__version__ = "0.1.0"


def __getattr__(name):
    """Lazy import to avoid loading UI when only using classification."""
    if name == "GeoVibes":
        from .ui import GeoVibes

        return GeoVibes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GeoVibes"]
