"""
ATLAS Phase 6.4 — Product Role & Capability Selector.

Evaluates product suitability and ranks candidate devices for semantic mission objectives.
Understands product roles (Vision, Glass, Drone, Rover), health, connectivity, battery,
and capability descriptors.

CRITICAL ARCHITECTURAL RULES:
1. NO DIRECT EXECUTION: Role selection only produces semantic recommendations.
2. MODEL NEUTRAL: Transport and hardware neutral evaluation.
3. ADAPTER EQUIVALENCE: Evaluates DeviceIdentity and DigitalTwinInterface identically.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.interfaces.mission_interface import ProductRoleSelectorInterface
from core.models.device_contract import DeviceHealthStatus, ProductRole, ProductType
from core.models.mission import MissionObjective, MissionObjectiveType
from core.models.orchestration import ConnectivityStatus, DeviceIdentity


class ProductRoleSelector(ProductRoleSelectorInterface):
    """
    Deterministic product suitability evaluator and ranker.
    """

    # Primary suitability mapping by objective type and product type
    ROLE_SUITABILITY: Dict[MissionObjectiveType, Dict[ProductType, float]] = {
        MissionObjectiveType.VERIFY_INCIDENT: {
            ProductType.DRONE: 0.95,
            ProductType.ROVER: 0.80,
            ProductType.VISION: 0.70,
            ProductType.GLASS: 0.65,
        },
        MissionObjectiveType.LOCATE_TARGET: {
            ProductType.DRONE: 0.90,
            ProductType.VISION: 0.85,
            ProductType.ROVER: 0.75,
            ProductType.GLASS: 0.60,
        },
        MissionObjectiveType.ASSESS_THREAT: {
            ProductType.DRONE: 0.90,
            ProductType.ROVER: 0.85,
            ProductType.VISION: 0.80,
            ProductType.GLASS: 0.70,
        },
        MissionObjectiveType.MAINTAIN_OBSERVATION: {
            ProductType.VISION: 0.95,
            ProductType.DRONE: 0.75,
            ProductType.ROVER: 0.70,
            ProductType.GLASS: 0.50,
        },
        MissionObjectiveType.NOTIFY_WEARER: {
            ProductType.GLASS: 0.99,
            ProductType.VISION: 0.10,
            ProductType.DRONE: 0.10,
            ProductType.ROVER: 0.10,
        },
        MissionObjectiveType.INSPECT_ROUTE: {
            ProductType.ROVER: 0.95,
            ProductType.DRONE: 0.85,
            ProductType.VISION: 0.50,
            ProductType.GLASS: 0.40,
        },
        MissionObjectiveType.DISPATCH_RESPONSE: {
            ProductType.DRONE: 0.90,
            ProductType.ROVER: 0.90,
            ProductType.GLASS: 0.40,
            ProductType.VISION: 0.20,
        },
        MissionObjectiveType.MONITOR_AREA: {
            ProductType.VISION: 0.95,
            ProductType.DRONE: 0.80,
            ProductType.ROVER: 0.75,
            ProductType.GLASS: 0.50,
        },
        MissionObjectiveType.CONFIRM_RESOLUTION: {
            ProductType.VISION: 0.90,
            ProductType.DRONE: 0.85,
            ProductType.ROVER: 0.80,
            ProductType.GLASS: 0.75,
        },
    }

    SEMANTIC_CAPABILITY_MAP: Dict[str, Tuple[str, ...]] = {
        "flight": ("takeoff", "land", "hover", "goto_location", "navigate_waypoint", "return_to_base"),
        "aerial": ("takeoff", "land", "hover", "goto_location", "navigate_waypoint"),
        "locomotion": ("move", "navigate_to", "patrol_zone", "dock", "goto_location"),
        "ground": ("move", "navigate_to", "patrol_zone", "dock"),
        "camera": ("capture_image", "capture_video", "detect_motion", "detect_person", "record_video"),
        "video": ("capture_video", "record_video", "start_recording", "stop_recording"),
        "image": ("capture_image", "snapshot"),
        "hud": ("display_hud", "show_hud"),
        "notification": ("send_notification", "display_hud"),
        "audio": ("capture_audio", "record_audio"),
        "navigation": ("goto_location", "navigate_to", "navigate_waypoint", "move"),
        "stationary_camera": ("capture_image", "capture_video", "emit_event"),
        "continuous_monitoring": ("capture_image", "emit_event", "patrol_zone"),
        "detection": ("detect_motion", "detect_person", "emit_event", "capture_image"),
    }

    def select_candidate_products(
        self,
        objective: MissionObjective,
        available_devices: Sequence[Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Sequence[str]:
        """
        Return ordered list of candidate device IDs best suited to execute the objective.
        Filters out offline, unhealthy, or depleted devices.
        """
        ranked = []
        for dev in available_devices:
            score = self._score_device_for_objective(dev, objective, constraints)
            if score > 0.0:
                dev_id = self._extract_device_id(dev)
                ranked.append((dev_id, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return tuple(r[0] for r in ranked)

    def rank_products_for_capability(
        self,
        capability: str,
        available_devices: Sequence[Any],
    ) -> Sequence[Tuple[str, float]]:
        """
        Rank devices by suitability for a specific capability.
        """
        ranked = []
        for dev in available_devices:
            dev_id = self._extract_device_id(dev)
            score = self._score_device_for_capability(dev, capability)
            if score > 0.0:
                ranked.append((dev_id, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return tuple(ranked)

    def _device_has_capability(self, ptype: ProductType, caps: Sequence[str], req: str) -> bool:
        req_lower = req.lower()
        # Direct substring match in device capability list
        if any(req_lower in c.lower() for c in caps):
            return True
        # Check semantic category map
        synonyms = self.SEMANTIC_CAPABILITY_MAP.get(req_lower, ())
        if any(any(syn in c.lower() for syn in synonyms) for c in caps):
            return True
        # Product type fallbacks
        if req_lower in ("flight", "aerial") and ptype == ProductType.DRONE:
            return True
        if req_lower in ("locomotion", "ground") and ptype in (ProductType.ROVER, ProductType.DRONE):
            return True
        if req_lower in ("hud", "wearer_notification") and ptype == ProductType.GLASS:
            return True
        if req_lower in ("stationary_camera", "fixed_surveillance") and ptype == ProductType.VISION:
            return True
        if req_lower in ("camera", "imaging", "detection") and ptype in (
            ProductType.VISION,
            ProductType.DRONE,
            ProductType.ROVER,
            ProductType.GLASS,
        ):
            return True
        return False

    def _score_device_for_objective(
        self,
        device: Any,
        objective: MissionObjective,
        constraints: Optional[Dict[str, Any]],
    ) -> float:
        """Compute suitability score [0.0, 1.0] of a device for an objective."""
        ptype = self._extract_product_type(device)
        conn = self._extract_connectivity(device)
        health = self._extract_health(device)
        battery = self._extract_battery(device)

        # Disconnected or unhealthy device cannot be selected
        if conn not in (ConnectivityStatus.ONLINE, "ONLINE"):
            return 0.0
        if health in (DeviceHealthStatus.UNHEALTHY, "UNHEALTHY"):
            return 0.0

        # Critical battery device (<15%) cannot execute travel missions
        if battery is not None and battery < 15.0 and objective.type != MissionObjectiveType.NOTIFY_WEARER:
            return 0.0

        # Base role suitability
        base_score = self.ROLE_SUITABILITY.get(objective.type, {}).get(ptype, 0.40)

        # Health penalty
        if health in (DeviceHealthStatus.DEGRADED, "DEGRADED"):
            base_score -= 0.20

        # Battery penalty for low battery (15% - 30%)
        if battery is not None and battery < 30.0:
            base_score -= 0.15

        # Capability matching
        if objective.required_capabilities:
            caps = self._extract_capabilities(device)
            match_count = sum(1 for req in objective.required_capabilities if self._device_has_capability(ptype, caps, req))
            if match_count == 0 and caps:
                return 0.0

        return round(max(0.0, min(1.0, base_score)), 3)

    def _score_device_for_capability(self, device: Any, capability: str) -> float:
        conn = self._extract_connectivity(device)
        health = self._extract_health(device)
        if conn not in (ConnectivityStatus.ONLINE, "ONLINE") or health in (DeviceHealthStatus.UNHEALTHY, "UNHEALTHY"):
            return 0.0

        ptype = self._extract_product_type(device)
        caps = self._extract_capabilities(device)
        if not self._device_has_capability(ptype, caps, capability):
            return 0.0

        score = 0.90
        if health in (DeviceHealthStatus.DEGRADED, "DEGRADED"):
            score -= 0.20
        return score

    # ========================================================================
    # Attribute Extractors
    # ========================================================================

    def _extract_device_id(self, device: Any) -> str:
        if hasattr(device, "twin_id"):
            return str(device.twin_id)
        if hasattr(device, "device_id"):
            return str(device.device_id)
        if isinstance(device, dict):
            return str(device.get("device_id", device.get("twin_id", "UNKNOWN")))
        return str(device)

    def _extract_product_type(self, device: Any) -> ProductType:
        if hasattr(device, "product_type"):
            pt = device.product_type
            return pt if isinstance(pt, ProductType) else ProductType.from_str(str(pt))
        if hasattr(device, "get_state"):
            return device.get_state().product_type
        if isinstance(device, dict):
            return ProductType.from_str(device.get("product_type", "UNKNOWN"))
        return ProductType.UNKNOWN

    def _extract_connectivity(self, device: Any) -> ConnectivityStatus:
        if hasattr(device, "get_state"):
            return device.get_state().connectivity
        if hasattr(device, "connectivity_status"):
            return device.connectivity_status
        if hasattr(device, "connectivity"):
            return device.connectivity
        return ConnectivityStatus.ONLINE

    def _extract_health(self, device: Any) -> DeviceHealthStatus:
        if hasattr(device, "_health"):
            return device._health
        if hasattr(device, "health"):
            return device.health
        if hasattr(device, "get_state"):
            return device.get_state().health
        return DeviceHealthStatus.HEALTHY

    def _extract_battery(self, device: Any) -> Optional[float]:
        if hasattr(device, "get_state"):
            return float(device.get_state().battery)
        if hasattr(device, "battery_pct"):
            return float(device.battery_pct)
        if hasattr(device, "battery"):
            return float(device.battery)
        return None

    def _extract_capabilities(self, device: Any) -> List[str]:
        if hasattr(device, "get_state"):
            return list(device.get_state().active_capabilities)
        if hasattr(device, "capabilities"):
            caps = device.capabilities
            if isinstance(caps, (list, tuple)):
                res = []
                for c in caps:
                    if hasattr(c, "capability_name"):
                        res.append(c.capability_name)
                    elif hasattr(c, "capability_id"):
                        res.append(c.capability_id)
                    else:
                        res.append(str(c))
                return res
        return []
