"""Status messaging utilities for the GeoVibes UI."""

from __future__ import annotations

from dataclasses import dataclass


_STATUS_TEMPLATE = """
<div style='background: white; padding: 5px; border-radius: 5px; opacity: 0.8; font-size: 12px;'>
    {body}
</div>
""".strip()


@dataclass
class StatusBus:
    """Centralizes status/operation messages displayed in the UI."""

    current_operation: str | None = None

    def set_operation(self, message: str) -> None:
        self.current_operation = message

    def clear_operation(self) -> None:
        self.current_operation = None

    def render(
        self,
        *,
        lat: float,
        lon: float,
        mode: str,
        label: str,
        polygon_drawing: bool = False,
    ) -> str:
        """Return HTML for the status bar."""
        base = f"Lat: {lat:.4f} | Lon: {lon:.4f} | Mode: {mode} | Label: {label}"
        if polygon_drawing:
            base += " | <b>Drawing polygon...</b>"

        if self.current_operation:
            base += (
                "<br/><span style='color: #0072B2; font-weight: bold;'>"
                f"{self.current_operation}</span>"
            )

        return _STATUS_TEMPLATE.format(body=base)


__all__ = ["StatusBus"]
