"""
ATLAS Phase 6.3 — Simulated Environment & Entities.

Represents spatial entities, environmental hazards, ambient weather/lighting,
and geographic bounds for deterministic multi-agent simulation.
"""

from dataclasses import dataclass, field
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.simulation import SimulationLimits, TwinPosition


@dataclass(frozen=True)
class SimulatedEntity:
    """
    Physical or semantic object present in the simulated world (e.g. human, vehicle, obstacle).
    """
    entity_id: str
    entity_type: str  # "PERSON", "VEHICLE", "OBSTACLE", "ANOMALY"
    position: TwinPosition
    velocity: float = 0.0  # m/s
    heading: float = 0.0   # degrees
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "position": self.position.to_dict(),
            "velocity": self.velocity,
            "heading": self.heading,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulatedEntity":
        return cls(
            entity_id=str(data["entity_id"]),
            entity_type=str(data.get("entity_type", "OBSTACLE")),
            position=TwinPosition.from_dict(data["position"]),
            velocity=float(data.get("velocity", 0.0)),
            heading=float(data.get("heading", 0.0)),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class SimulatedHazard:
    """
    Environmental hazard or sensory trigger (e.g. motion, smoke, heat anomaly, restricted zone).
    """
    hazard_id: str
    hazard_type: str  # "MOTION", "SMOKE", "HEAT_ANOMALY", "RESTRICTED_ZONE"
    location: TwinPosition
    radius_meters: float = 10.0
    severity: float = 0.8
    is_active: bool = True
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "hazard_type": self.hazard_type,
            "location": self.location.to_dict(),
            "radius_meters": self.radius_meters,
            "severity": self.severity,
            "is_active": self.is_active,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulatedHazard":
        return cls(
            hazard_id=str(data["hazard_id"]),
            hazard_type=str(data.get("hazard_type", "MOTION")),
            location=TwinPosition.from_dict(data["location"]),
            radius_meters=float(data.get("radius_meters", 10.0)),
            severity=float(data.get("severity", 0.8)),
            is_active=bool(data.get("is_active", True)),
            attributes=dict(data.get("attributes", {})),
        )


class SimulatedEnvironment:
    """
    Container managing spatial boundaries, ambient environmental properties,
    simulated entities, and hazards with strict capacity bounds.
    """

    def __init__(
        self,
        ambient_temperature_celsius: float = 22.0,
        weather: str = "CLEAR",
        lighting_lux: float = 500.0,
        limits: Optional[SimulationLimits] = None,
    ):
        self._lock = threading.RLock()
        self.ambient_temperature_celsius = ambient_temperature_celsius
        self.weather = weather
        self.lighting_lux = lighting_lux
        self.limits = limits or SimulationLimits()

        self._entities: Dict[str, SimulatedEntity] = {}
        self._hazards: Dict[str, SimulatedHazard] = {}

    def add_entity(self, entity: SimulatedEntity) -> None:
        with self._lock:
            if len(self._entities) >= self.limits.max_entities and entity.entity_id not in self._entities:
                raise ValueError(f"Simulation entity capacity bound ({self.limits.max_entities}) exceeded")
            self._entities[entity.entity_id] = entity

    def remove_entity(self, entity_id: str) -> bool:
        with self._lock:
            return self._entities.pop(entity_id, None) is not None

    def get_entity(self, entity_id: str) -> Optional[SimulatedEntity]:
        with self._lock:
            return self._entities.get(entity_id)

    def list_entities(self) -> List[SimulatedEntity]:
        with self._lock:
            return list(self._entities.values())

    def add_hazard(self, hazard: SimulatedHazard) -> None:
        with self._lock:
            if len(self._hazards) >= self.limits.max_entities and hazard.hazard_id not in self._hazards:
                raise ValueError("Simulation hazard capacity bound exceeded")
            self._hazards[hazard.hazard_id] = hazard

    def remove_hazard(self, hazard_id: str) -> bool:
        with self._lock:
            return self._hazards.pop(hazard_id, None) is not None

    def get_hazard(self, hazard_id: str) -> Optional[SimulatedHazard]:
        with self._lock:
            return self._hazards.get(hazard_id)

    def list_hazards(self) -> List[SimulatedHazard]:
        with self._lock:
            return list(self._hazards.values())

    def get_entities_near(self, position: TwinPosition, radius_meters: float) -> List[SimulatedEntity]:
        """Find all entities within radius_meters of position."""
        with self._lock:
            results = []
            for entity in self._entities.values():
                if position.distance_to(entity.position) <= radius_meters:
                    results.append(entity)
            return results

    def get_hazards_near(self, position: TwinPosition, radius_meters: float) -> List[SimulatedHazard]:
        """Find all active hazards within radius_meters + hazard.radius_meters of position."""
        with self._lock:
            results = []
            for hazard in self._hazards.values():
                if not hazard.is_active:
                    continue
                effective_radius = radius_meters + hazard.radius_meters
                if position.distance_to(hazard.location) <= effective_radius:
                    results.append(hazard)
            return results

    def step(self, delta_seconds: float) -> None:
        """
        Advance simulated entities by delta_seconds based on velocity and heading.
        """
        with self._lock:
            updated_entities = {}
            for entity_id, entity in self._entities.items():
                if entity.velocity <= 0.0:
                    updated_entities[entity_id] = entity
                    continue

                dist = entity.velocity * delta_seconds
                rad = math.radians(entity.heading)
                dlat = (dist * math.cos(rad)) / 111000.0
                dlon = (dist * math.sin(rad)) / (111000.0 * math.cos(math.radians(entity.position.latitude)))
                new_pos = TwinPosition(
                    latitude=entity.position.latitude + dlat,
                    longitude=entity.position.longitude + dlon,
                    altitude=entity.position.altitude,
                    heading=entity.heading,
                    speed=entity.velocity,
                )
                updated_entities[entity_id] = SimulatedEntity(
                    entity_id=entity.entity_id,
                    entity_type=entity.entity_type,
                    position=new_pos,
                    velocity=entity.velocity,
                    heading=entity.heading,
                    attributes=entity.attributes,
                )
            self._entities = updated_entities

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ambient_temperature_celsius": self.ambient_temperature_celsius,
                "weather": self.weather,
                "lighting_lux": self.lighting_lux,
                "entities": [e.to_dict() for e in self._entities.values()],
                "hazards": [h.to_dict() for h in self._hazards.values()],
            }
