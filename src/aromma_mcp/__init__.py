"""AROMMA FastMCP package."""

from aromma_mcp.server import (
    mcp,
    predict_odor,
    recommend_molecules,
    modulate_scent,
    list_descriptors,
)

__version__ = "0.2.0"
__all__ = [
    "mcp",
    "predict_odor",
    "recommend_molecules",
    "modulate_scent",
    "list_descriptors",
]
