# mazedemo/ids.py
"""
Defines typed, hashable identifiers for surfaces and their elements to
improve type safety and clarity throughout the library.
"""

from dataclasses import dataclass
from typing import Hashable, Optional
import uuid
import numpy as np


@dataclass(frozen=True, eq=True)
class SurfaceId:
    """
    A unique, persistent identifier for a Surface.
    The UUID is a placeholder until content-based hashing is implemented.
    """

    _uuid: uuid.UUID
    _hash: Optional[bytes] = None  # To be used for content-based hashing

    def __hash__(self):
        # Manually implement hash caching for this frozen dataclass.
        if not hasattr(self, "_hash_cache"):
            # The default hash for a frozen dataclass with eq=True is based on its fields.
            h = hash((self._uuid, self._hash))
            object.__setattr__(self, "_hash_cache", h)
        return self._hash_cache

    def __str__(self) -> str:
        return str(self._uuid)


@dataclass(frozen=True)
class ElementId:
    """Base class for all element identifiers, containing the core data."""

    _surface_id: SurfaceId
    _local_id: int

    def __eq__(self, other):
        if not isinstance(other, ElementId):
            return NotImplemented
        return (
            type(self) is type(other)
            and self._surface_id == other._surface_id
            and self._local_id == other._local_id
        )

    def __hash__(self):
        # Manually implement hash caching for this frozen dataclass.
        if not hasattr(self, "_hash_cache"):
            h = hash((type(self), self._surface_id, self._local_id))
            object.__setattr__(self, "_hash_cache", h)
        return self._hash_cache

    def __lt__(self, other_id):
        if not isinstance(other_id, ElementId):
            raise ValueError("ids are only comparable with other ids")
        if self._surface_id != other_id._surface_id:
            raise ValueError("ids from different surfaces are not comparable")
        else:
            return self._local_id < other_id._local_id

    @property
    def surface_id(self):
        return self._surface_id

    @property
    def local_id(self) -> int:
        return self._local_id

    @property
    def id(self) -> int:
        return self._local_id


def get_local_id(fid: Hashable) -> int:
    """Safely extracts integer local ID from FaceId, ElementId, int, or string."""
    if hasattr(fid, "local_id"):
        return int(fid.local_id)
    if hasattr(fid, "id"):
        return int(fid.id)
    if hasattr(fid, "_local_id"):
        return int(getattr(fid, "_local_id"))
    if isinstance(fid, (int, np.integer)):
        return int(fid)
    if isinstance(fid, str):
        if fid.startswith("F") or fid.startswith("V") or fid.startswith("HE"):
            return int(fid[1:].strip("()"))
        return int(fid)
    return int(fid)


class VertexId(ElementId):
    """A type-safe wrapper for a vertex identifier."""

    def __str__(self):
        return f"V({self._local_id})"


class FaceId(ElementId):
    """A type-safe wrapper for a face identifier."""

    def __str__(self):
        return f"F({self._local_id})"


class HalfEdgeId(ElementId):
    """A type-safe wrapper for a half-edge identifier."""

    def __str__(self):
        return f"HE({self._local_id})"
