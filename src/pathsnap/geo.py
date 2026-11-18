"""Geographic helpers.

Each trip covers at most a few tens of kilometres, so we project lon/lat to a
local azimuthal-equidistant-like tangent plane (equirectangular about the trip
centroid). Over < 50 km the distance error versus the geodesic is far below
GPS noise (< 0.1 %); this is verified in scripts/01_eda.py against haversine.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6_371_008.8  # mean Earth radius (m)


class LocalProjector:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.deg2rad(1.0) * R_EARTH * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * R_EARTH

    @classmethod
    def for_points(cls, lonlat: np.ndarray) -> "LocalProjector":
        return cls(np.mean(lonlat[:, 0]), np.mean(lonlat[:, 1]))

    def fwd(self, lonlat: np.ndarray) -> np.ndarray:
        x = (lonlat[:, 0] - self.lon0) * self.kx
        y = (lonlat[:, 1] - self.lat0) * self.ky
        return np.column_stack([x, y])

    def inv(self, xy: np.ndarray) -> np.ndarray:
        return np.column_stack([xy[:, 0] / self.kx + self.lon0,
                                xy[:, 1] / self.ky + self.lat0])


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised point-to-segment projection.

    p, a, b: (..., 2) broadcastable. Returns (distance, t_clamped, t_raw)
    where t is the fractional position of the projection along a->b.
    """
    ab = b - a
    ap = p - a
    L2 = np.einsum("...i,...i->...", ab, ab)
    t_raw = np.einsum("...i,...i->...", ap, ab) / np.maximum(L2, 1e-9)
    t = np.clip(t_raw, 0.0, 1.0)
    proj = a + t[..., None] * ab
    d = np.linalg.norm(p - proj, axis=-1)
    return d, t, t_raw


def bearing_xy(dx, dy):
    """Compass bearing in degrees [0, 360) of a vector in the local plane."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d
