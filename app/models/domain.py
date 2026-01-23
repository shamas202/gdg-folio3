from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    category: str
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2
    score: float
    mask_polygon: list | None = None  # Polygon coordinates from RF-DETR API


@dataclass(frozen=True)
class Segment:
    bbox: tuple[int, int, int, int]
    mask: "object | None"  # keep generic; could be np.ndarray