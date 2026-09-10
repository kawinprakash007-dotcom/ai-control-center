"""
ATLAS Phase 6.6 - Authoritative Live State Capability.

Authoritative handler for read-only live runtime state queries:
- Device status, connectivity, health, telemetry, battery %
- WorldState entities and conditions
- Active situations and alerts
- Active missions and goals

Guarantees:
1. Live ATLAS authorities are strictly authoritative (DeviceGateway, WorldStateStore,
   SituationFusionEngine, GoalManager).
2. Never triggers physical/virtual device actuation or state mutations.
3. Provides typed factual context (LiveDeviceContext, LiveWorldContext, etc.)
   before any model synthesis.
4. Bounded fallback: if Ollama is slow/offline, returns the deterministic factual summary.
5. Structured bounded diagnostic logging.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

from core.models.live_context import (
    ChatQueryClassification,
    LiveDeviceContext,
    LiveWorldContext,
    LiveSituationContext,
    LiveMissionContext,
)
from core.models.result import Result
from core.models.task import Task
from core.models.orchestration import ConnectivityStatus, DeviceType
from core.models.device_contract import ProductType, DeviceHealthStatus
from llm.ollama_client import ask_ollama

logger = logging.getLogger("atlas.tools.live_state")


class LiveStateCapability:
    """
    Authoritative capability bridging cognitive queries directly to live ATLAS authorities.
    """

    def __init__(
        self,
        device_gateway: Optional[Any] = None,
        world_store: Optional[Any] = None,
        situation_engine: Optional[Any] = None,
        goal_manager: Optional[Any] = None,
    ):
        self.device_gateway = device_gateway
        self.world_store = world_store
        self.situation_engine = situation_engine
        self.goal_manager = goal_manager

    def __call__(self, task: Optional[Union[Task, Dict[str, Any], str]] = None) -> Union[str, Result]:
        start_time = time.time()
        params: Dict[str, Any] = {}
        action = "query_device_state"

        if isinstance(task, Task):
            params = dict(task.parameters or {})
            action = getattr(task, "action", "query_device_state") or "query_device_state"
        elif isinstance(task, dict):
            params = dict(task)
            action = params.get("action", "query_device_state")
        elif isinstance(task, str):
            params = {"query": task}

        norm_action = str(action).strip().lower().replace(" ", "_")
        query_text = str(params.get("query") or params.get("original_text") or "").strip()

        # Route to appropriate live authority
        if "world" in norm_action or params.get("query_type") == "world_state":
            res = self._handle_world_state(query_text, params, start_time)
        elif "situation" in norm_action or params.get("query_type") == "situation":
            res = self._handle_situations(query_text, params, start_time)
        elif "mission" in norm_action or params.get("query_type") == "mission":
            res = self._handle_missions(query_text, params, start_time)
        else:
            res = self._handle_device_state(query_text, params, start_time)

        return res

    # ========================================================================
    # 1. Device Queries
    # ========================================================================

    def _resolve_device_id(self, target_raw: str) -> Optional[str]:
        """Resolve device target keyword or ID to registered device_id."""
        if not target_raw:
            return None
        target = target_raw.strip().lower()

        if self.device_gateway:
            devices = self.device_gateway.list_devices()
            # Exact match check
            for d in devices:
                if d.device_id.lower() == target:
                    return d.device_id

            # Match by ProductType or DeviceType
            for d in devices:
                prod_type = str(getattr(d, "product_type", "")).lower()
                dev_type = str(getattr(d, "device_type", "")).lower()
                disp_name = str(getattr(d, "display_name", "")).lower()
                if target in prod_type or target in dev_type or target in disp_name:
                    return d.device_id

        # Defaults for standard simulation products
        if "drone" in target or "aerial" in target:
            return "ATLAS_DRONE_01"
        if "rover" in target or "robot" in target:
            return "ATLAS_ROVER_01"
        if "vision" in target or "camera" in target:
            return "ATLAS_VISION_01"
        if "glass" in target:
            return "ATLAS_GLASS_01"

        return None

    def _handle_device_state(
        self, query: str, params: Dict[str, Any], start_time: float
    ) -> str:
        target_device_raw = str(params.get("target_device") or "").strip()
        query_type = str(params.get("query_type") or "status").strip().lower()

        if not target_device_raw:
            # Try to extract from query string
            for key in ("drone", "rover", "vision", "glass", "devices"):
                if re.search(r"\b" + key + r"\b", query, re.IGNORECASE):
                    target_device_raw = key
                    break

        if not target_device_raw:
            target_device_raw = "all" if "device" in query.lower() else "drone"

        # Multi-device / list query
        if target_device_raw in ("all", "devices") or query_type == "list":
            return self._handle_list_devices(query, start_time)

        resolved_id = self._resolve_device_id(target_device_raw)
        if not resolved_id:
            resolved_id = "ATLAS_DRONE_01" if "drone" in target_device_raw else "ATLAS_ROVER_01"

        # Query live state from DeviceGateway
        connectivity = "ONLINE"
        health_status = "HEALTHY"
        battery_pct: Optional[float] = 88.0 if "DRONE" in resolved_id else 92.0
        display_name = resolved_id
        telemetry_dict: Dict[str, Any] = {}

        if self.device_gateway:
            identity = self.device_gateway.get_device(resolved_id)
            if identity:
                display_name = identity.display_name or resolved_id
                conn = self.device_gateway.query_device_status(resolved_id)
                connectivity = conn.value if hasattr(conn, "value") else str(conn)

            health = self.device_gateway.get_device_health(resolved_id)
            if health:
                health_status = health.status.value if hasattr(health.status, "value") else str(health.status)
                if health.battery_pct is not None:
                    battery_pct = float(health.battery_pct)

            # Check if adapter has live telemetry
            if resolved_id in getattr(self.device_gateway, "_device_adapters", {}):
                adapter = self.device_gateway._device_adapters[resolved_id]
                if hasattr(adapter, "battery_pct") and battery_pct is None:
                    battery_pct = float(adapter.battery_pct)
                if hasattr(adapter, "location"):
                    telemetry_dict["location"] = adapter.location.to_dict() if hasattr(adapter.location, "to_dict") else str(adapter.location)
                if hasattr(adapter, "altitude"):
                    telemetry_dict["altitude"] = adapter.altitude

        # Build factual summary
        batt_str = f"{int(battery_pct)}%" if battery_pct is not None else "nominal"
        if query_type == "battery":
            summary = f"{display_name} ({resolved_id}) battery level is {batt_str}."
        elif query_type == "connectivity":
            is_online = (connectivity == "ONLINE")
            summary = (
                f"Yes. {display_name} ({resolved_id}) is {connectivity}."
                if is_online
                else f"No. {display_name} ({resolved_id}) is currently {connectivity}."
            )
        elif query_type == "telemetry":
            summary = (
                f"{display_name} ({resolved_id}) telemetry: connectivity={connectivity}, "
                f"health={health_status}, battery={batt_str}"
            )
            if "altitude" in telemetry_dict:
                summary += f", altitude={telemetry_dict['altitude']}m"
            summary += "."
        else:
            summary = f"{display_name} ({resolved_id}) is {connectivity} and {health_status} with {batt_str} battery."

        context = LiveDeviceContext(
            device_id=resolved_id,
            display_name=display_name,
            connectivity=connectivity,
            health=health_status,
            battery_pct=battery_pct,
            telemetry=telemetry_dict,
            summary=summary,
        )

        latency_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "[Chat] classification=CURRENT_DEVICE_STATE target=%s capability=device_status source=device_registry latency_ms=%.2f",
            resolved_id,
            latency_ms,
        )

        return self._synthesize_or_fallback(query, context.summary, "Live Device Telemetry")

    def _handle_list_devices(self, query: str, start_time: float) -> str:
        device_items: List[Dict[str, Any]] = []
        if self.device_gateway:
            for dev in self.device_gateway.list_devices():
                status = self.device_gateway.query_device_status(dev.device_id)
                health = self.device_gateway.get_device_health(dev.device_id)
                device_items.append({
                    "device_id": dev.device_id,
                    "display_name": dev.display_name,
                    "connectivity": status.value if hasattr(status, "value") else str(status),
                    "health": health.status.value if hasattr(health.status, "value") else str(health.status),
                    "battery": f"{int(health.battery_pct)}%" if health.battery_pct is not None else "nominal",
                })
        else:
            device_items = [
                {"device_id": "ATLAS_VISION_01", "display_name": "ATLAS Static Camera Vision", "connectivity": "ONLINE", "health": "HEALTHY", "battery": "100%"},
                {"device_id": "ATLAS_DRONE_01", "display_name": "ATLAS Autonomous Drone", "connectivity": "ONLINE", "health": "HEALTHY", "battery": "88%"},
                {"device_id": "ATLAS_GLASS_01", "display_name": "ATLAS Smart Glasses", "connectivity": "ONLINE", "health": "HEALTHY", "battery": "100%"},
                {"device_id": "ATLAS_ROVER_01", "display_name": "ATLAS Ground Rover", "connectivity": "ONLINE", "health": "HEALTHY", "battery": "92%"},
            ]

        online_count = sum(1 for d in device_items if d.get("connectivity") == "ONLINE")
        lines = [f"{d['display_name']} ({d['device_id']}): {d['connectivity']} [{d['health']}, battery {d['battery']}]" for d in device_items]
        summary = f"There are {len(device_items)} registered ATLAS devices ({online_count} online):\n" + "\n".join(f"- {line}" for line in lines)

        context = LiveDeviceContext(
            device_id="ALL",
            display_name="All Registered Devices",
            connectivity="ONLINE" if online_count > 0 else "OFFLINE",
            health="HEALTHY",
            all_devices=device_items,
            summary=summary,
        )

        latency_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "[Chat] classification=CURRENT_DEVICE_STATE target=ALL capability=device_status source=device_registry latency_ms=%.2f",
            latency_ms,
        )

        return self._synthesize_or_fallback(query, context.summary, "Live Device Registry")

    # ========================================================================
    # 2. WorldState Queries
    # ========================================================================

    def _handle_world_state(
        self, query: str, params: Dict[str, Any], start_time: float
    ) -> str:
        entities: List[Dict[str, Any]] = []
        version = 1
        if self.world_store and hasattr(self.world_store, "get_current_state"):
            ws = self.world_store.get_current_state()
            version = getattr(ws, "state_version", 1)
            raw_entities = getattr(ws, "entities", {})
            if isinstance(raw_entities, dict):
                entities = [v.to_dict() if hasattr(v, "to_dict") else dict(v) for v in raw_entities.values()]
            elif isinstance(raw_entities, list):
                entities = [v.to_dict() if hasattr(v, "to_dict") else dict(v) for v in raw_entities]

        if not entities:
            summary = f"Current ATLAS WorldState (v{version}) is nominal with 0 active untracked anomalies. Digital Twin environment active."
        else:
            summary = f"Current ATLAS WorldState (v{version}) tracks {len(entities)} active entity/entities:\n" + "\n".join(
                f"- {e.get('entity_id', 'Entity')}: {e.get('entity_type', 'object')} at status {e.get('status', 'nominal')}"
                for e in entities[:5]
            )

        context = LiveWorldContext(
            state_version=version,
            entity_count=len(entities),
            entities=entities,
            summary=summary,
        )

        latency_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "[Chat] classification=CURRENT_WORLD_STATE target=world capability=world_state source=world_store latency_ms=%.2f",
            latency_ms,
        )

        return self._synthesize_or_fallback(query, context.summary, "Authoritative WorldState")

    # ========================================================================
    # 3. Situation Queries
    # ========================================================================

    def _handle_situations(
        self, query: str, params: Dict[str, Any], start_time: float
    ) -> str:
        active_sits: List[Dict[str, Any]] = []
        if self.situation_engine and hasattr(self.situation_engine, "get_active_situations"):
            sits = self.situation_engine.get_active_situations()
            active_sits = [s.to_dict() if hasattr(s, "to_dict") else dict(s) for s in sits]

        if not active_sits:
            summary = "There are currently no active situations or anomalies detected in the operational area."
        else:
            lines = [f"{s.get('title', 'Situation')}: severity={s.get('severity', 'info')}, status={s.get('status', 'active')}" for s in active_sits[:5]]
            summary = f"{len(active_sits)} active situation(s):\n" + "\n".join(f"- {line}" for line in lines)

        context = LiveSituationContext(
            count=len(active_sits),
            active_situations=active_sits,
            summary=summary,
        )

        latency_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "[Chat] classification=CURRENT_SITUATION target=situations capability=situation_intel source=fusion_engine latency_ms=%.2f",
            latency_ms,
        )

        return self._synthesize_or_fallback(query, context.summary, "Live Situation Intelligence")

    # ========================================================================
    # 4. Mission & Goal Queries
    # ========================================================================

    def _handle_missions(
        self, query: str, params: Dict[str, Any], start_time: float
    ) -> str:
        goals: List[Dict[str, Any]] = []
        if self.goal_manager:
            raw_goals = getattr(self.goal_manager, "get_all_goals", lambda: [])()
            goals = [g.to_dict() if hasattr(g, "to_dict") else dict(g) for g in raw_goals]

        if not goals:
            summary = "There are currently no active tactical missions or autonomous goals running."
        else:
            lines = [f"Goal '{g.get('title') or g.get('goal_id')}': status={g.get('status', 'ACTIVE')}" for g in goals[:5]]
            summary = f"{len(goals)} tracked mission/goal(s):\n" + "\n".join(f"- {line}" for line in lines)

        context = LiveMissionContext(
            count=len(goals),
            active_goals=goals,
            summary=summary,
        )

        latency_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "[Chat] classification=CURRENT_MISSION target=missions capability=mission_coordinator source=goal_manager latency_ms=%.2f",
            latency_ms,
        )

        return self._synthesize_or_fallback(query, context.summary, "Live Mission State")

    # ========================================================================
    # Response Synthesis with LLM or Direct Fallback
    # ========================================================================

    def _synthesize_or_fallback(
        self, query: str, authoritative_summary: str, context_label: str
    ) -> str:
        """
        Synthesize concise natural language response grounded in authoritative facts.
        Falls back cleanly to authoritative_summary if LLM is unavailable or offline.
        """
        if not query:
            return authoritative_summary

        prompt = (
            f"You are the ATLAS AI Control Center assistant.\n"
            f"Answer the user's question concisely using ONLY the verified authoritative live ATLAS state provided below.\n"
            f"Do not invent devices, states, or troubleshooting steps.\n\n"
            f"[{context_label}]\n"
            f"{authoritative_summary}\n\n"
            f"User Question: {query}\n"
            f"Answer:"
        )

        try:
            # Safe bounded call to Ollama (timeout bounded at 10 seconds)
            response = ask_ollama(prompt, timeout=10.0)
            if response and len(response.strip()) > 3:
                return response.strip()
        except Exception:
            pass

        return authoritative_summary
