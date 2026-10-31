# mazedemo/surface_property_manager.py
"""
Implements the SurfacePropertyManager for handling data associated with
surfaces and their elements, decoupled from the topology.
"""

from typing import Any, Dict, Hashable


class SurfacePropertyManager:
    """
    A centralized repository for all data associated with mesh elements,
    completely decoupled from the topological structures.
    """

    def __init__(self):
        self.vertex_properties: Dict[Hashable, Dict[str, Any]] = {}
        self.face_properties: Dict[Hashable, Dict[str, Any]] = {}
        self.edge_properties: Dict[Hashable, Dict[str, Any]] = {}
        self.half_edge_properties: Dict[Hashable, Dict[str, Any]] = {}

    def _get_property_dict(self, element_type: str) -> Dict[Hashable, Dict[str, Any]]:
        """Helper to get the correct property dictionary based on element type."""
        if element_type == "vertex":
            return self.vertex_properties
        elif element_type == "face":
            return self.face_properties
        elif element_type == "half_edge":
            return self.half_edge_properties
        # Add other element types as needed
        else:
            raise ValueError(f"Unknown element type: {element_type}")

    def set_property(
        self,
        element_type: str,
        element_id: Hashable,
        prop_name: str,
        value: Any,
        surface_id_for_key: Hashable = None,
    ):
        """
        Sets a property for a given mesh element.

        Args:
            element_type (str): 'vertex', 'face', etc.
            element_id (Hashable): The unique ID of the element.
            prop_name (str): The name of the property to set.
            value (Any): The value of the property.
            surface_id_for_key: If provided, the property is stored under this
                                SurfaceId, which is used for tombstones.
        """
        prop_dict = self._get_property_dict(element_type)
        key = element_id
        if surface_id_for_key:
            # For tombstones, the key is a tuple of the new surface and the old element
            key = (surface_id_for_key, element_id)

        if key not in prop_dict:
            prop_dict[key] = {}
        prop_dict[key][prop_name] = value

    def get_property(
        self,
        element_type: str,
        element_id: Hashable,
        prop_name: str,
        default: Any = None,
    ) -> Any:
        """
        Gets a property for a given mesh element.
        """
        prop_dict = self._get_property_dict(element_type)
        return prop_dict.get(element_id, {}).get(prop_name, default)
