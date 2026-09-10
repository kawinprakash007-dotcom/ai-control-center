"""
ATLAS Phase 6.6 - Authoritative Live Context Contracts & Query Classification.

Provides typed models for factual context handoff to reasoning/response composition,
ensuring live ATLAS authorities (DeviceGateway, WorldState, Situations, Missions)
are strictly authoritative over RAG and Web for current runtime state.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional


class ChatQueryClassification(str, Enum):
    """Structured query classification for ATLAS chat requests."""
    CURRENT_DEVICE_STATE = "CURRENT_DEVICE_STATE"
    CURRENT_WORLD_STATE = "CURRENT_WORLD_STATE"
    CURRENT_SITUATION = "CURRENT_SITUATION"
    CURRENT_MISSION = "CURRENT_MISSION"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    WEB_RESEARCH = "WEB_RESEARCH"
    ACTION_REQUEST = "ACTION_REQUEST"
    CONVERSATION = "CONVERSATION"

    @classmethod
    def from_str(cls, val: Any) -> "ChatQueryClassification":
        if isinstance(val, cls):
            return val
        s = str(val or "").strip().upper()
        for member in cls:
            if member.value == s or member.name == s:
                return member
        return cls.CONVERSATION


@dataclass(frozen=True)
class LiveDeviceContext:
    """Typed factual context representing authoritative live device telemetry/status."""
    device_id: str
    product_type: str = "UNKNOWN"
    display_name: str = ""
    connectivity: str = "UNKNOWN"
    health: str = "UNKNOWN"
    battery_pct: Optional[float] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)
    all_devices: Optional[List[Dict[str, Any]]] = None
    timestamp: float = field(default_factory=time.time)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "product_type": self.product_type,
            "display_name": self.display_name,
            "connectivity": self.connectivity,
            "health": self.health,
            "battery_pct": self.battery_pct,
            "telemetry": dict(self.telemetry),
            "all_devices": list(self.all_devices) if self.all_devices is not None else None,
            "timestamp": self.timestamp,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class LiveWorldContext:
    """Typed factual context representing authoritative ATLAS WorldState."""
    state_version: int = 0
    entity_count: int = 0
    entities: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_version": self.state_version,
            "entity_count": self.entity_count,
            "entities": list(self.entities),
            "conditions": dict(self.conditions),
            "timestamp": self.timestamp,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class LiveSituationContext:
    """Typed factual context representing authoritative active situations."""
    count: int = 0
    active_situations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "active_situations": list(self.active_situations),
            "timestamp": self.timestamp,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class LiveMissionContext:
    """Typed factual context representing authoritative active missions and goals."""
    count: int = 0
    active_missions: List[Dict[str, Any]] = field(default_factory=list)
    active_goals: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "active_missions": list(self.active_missions),
            "active_goals": list(self.active_goals),
            "timestamp": self.timestamp,
            "summary": self.summary,
        }
