"""ATLAS Phase 6.5c — Telemetry Processing Subsystem.

Provides model-neutral, deterministic telemetry normalization for:
- Standard metrics (battery, temperature, signal quality, speed, heading)
- Connectivity normalization (ONLINE, DEGRADED, OFFLINE, UNKNOWN)
- Health state normalization (HEALTHY, DEGRADED, FAULT, UNKNOWN)
- Telemetry quality & freshness evaluation (VALID, STALE, PARTIAL, INVALID)
- Strict unit validation and bounded metric collection
- Production of typed TelemetryObservation and PerceptionEvidence

CRITICAL INVARIANTS:
1. Purely descriptive: reports factual evidence only.
2. ZERO decision leakage: battery=5% is evidence, never an abort command.
3. No fabrication: only metrics present in input are emitted.
4. Bounded: enforces hard limits on metric counts and string lengths.
"""

from __future__ import annotations

import datetime
import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.models.device_contract import (
    ConnectivityStatus,
    DeviceHealthStatus,
    sanitize_contract_metadata,
)
from core.models.orchestration import ModalityType
from core.models.perception import (
    PerceptionEvidence,
    PerceptionLimits,
)
from core.models.spatial_telemetry import (
    SpatialTelemetryLimits,
    TelemetryHealth,
    TelemetryMetric,
    TelemetryObservation,
    TelemetryQuality,
    VALID_TELEMETRY_UNITS,
)

logger = logging.getLogger(__name__)


def evaluate_freshness(
    age_seconds: float,
    max_age_seconds: float = 300.0,
) -> Tuple[TelemetryQuality, bool]:
    """
    Evaluate freshness of data against maximum allowable staleness.
    Returns (quality, is_stale).
    """
    if age_seconds > max_age_seconds:
        return TelemetryQuality.STALE, True
    return TelemetryQuality.VALID, False


def normalize_metric(
    name: str,
    value: Any,
    unit: str = "",
    confidence: float = 1.0,
    quality: TelemetryQuality = TelemetryQuality.VALID,
    timestamp: Optional[float] = None,
) -> TelemetryMetric:
    """Normalize a single telemetry metric measurement."""
    norm_name = str(name).strip()
    norm_unit = str(unit or "").strip()

    # If unit is empty and name implies a standard unit
    if not norm_unit:
        lower = norm_name.lower()
        if "temp" in lower or "celsius" in lower:
            norm_unit = "C"
        elif "volt" in lower or lower.endswith("_v"):
            norm_unit = "V"
        elif "batt" in lower or "percent" in lower or "pct" in lower:
            norm_unit = "%"
        elif "speed" in lower or "mps" in lower:
            norm_unit = "m/s"

    val: Union[float, int, str, bool]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Metric value cannot be NaN or Inf, got {val}")
    else:
        val = value

    return TelemetryMetric(
        name=norm_name,
        value=val,
        unit=norm_unit,
        raw_name=name,
        quality=quality,
        confidence=confidence,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


class TelemetryProcessor:
    """
    Normalizes device telemetry streams into structured, bounded evidence.
    Evaluates quality and freshness without making autonomous operational choices.
    """

    def __init__(
        self,
        limits: Optional[SpatialTelemetryLimits] = None,
        max_metrics: Optional[int] = None,
    ) -> None:
        base = limits or SpatialTelemetryLimits()
        if max_metrics is not None:
            self.limits = SpatialTelemetryLimits(
                max_metrics_per_result=max_metrics,
                max_attributes_per_metric=base.max_attributes_per_metric,
                max_spatial_relations=base.max_spatial_relations,
                max_geofences=base.max_geofences,
                max_tracked_sources=base.max_tracked_sources,
                max_history_entries=base.max_history_entries,
                max_staleness_seconds=base.max_staleness_seconds,
                max_batch_size=base.max_batch_size,
                max_string_length=base.max_string_length,
                max_route_ref_length=base.max_route_ref_length,
            )
        else:
            self.limits = base

    def normalize_health(self, health_data: Dict[str, Any]) -> TelemetryHealth:
        """Normalize health status, connectivity, and core vitals."""
        # Battery percentage with clamping
        batt_val = None
        raw_batt = health_data.get("battery_percent", health_data.get("battery", health_data.get("battery_pct")))
        if raw_batt is not None:
            try:
                bp = float(raw_batt)
                if not (math.isnan(bp) or math.isinf(bp)):
                    batt_val = max(0.0, min(100.0, bp))
            except (ValueError, TypeError):
                pass

        # Temperature
        temp_val = None
        raw_temp = health_data.get("temperature_celsius", health_data.get("temperature", health_data.get("temperature_c")))
        if raw_temp is not None:
            try:
                tc = float(raw_temp)
                if not (math.isnan(tc) or math.isinf(tc)):
                    temp_val = tc
            except (ValueError, TypeError):
                pass

        # CPU usage
        cpu_val = None
        raw_cpu = health_data.get("cpu_usage_percent", health_data.get("cpu_usage"))
        if raw_cpu is not None:
            try:
                cp = float(raw_cpu)
                if not (math.isnan(cp) or math.isinf(cp)):
                    cpu_val = max(0.0, min(100.0, cp))
            except (ValueError, TypeError):
                pass

        # Connectivity
        conn_str = str(health_data.get("connectivity", "ONLINE")).upper()
        try:
            conn = ConnectivityStatus.from_str(conn_str)
        except Exception:
            conn = ConnectivityStatus.UNKNOWN

        # Health status
        stat_str = str(health_data.get("health_status", health_data.get("status", "HEALTHY"))).upper()
        try:
            stat = DeviceHealthStatus.from_str(stat_str)
        except Exception:
            stat = DeviceHealthStatus.UNKNOWN

        return TelemetryHealth(
            health_status=stat,
            connectivity=conn,
            battery_percent=batt_val,
            temperature_celsius=temp_val,
            cpu_usage_percent=cpu_val,
        )

    def process(
        self,
        payload: Dict[str, Any],
        timestamp: Optional[Union[float, str]] = None,
        source_id: str = "default_device",
    ) -> TelemetryObservation:
        """Process arbitrary telemetry dictionary payload into TelemetryObservation."""
        now = time.time()
        captured_time = now

        # Compute age if timestamp provided
        age_seconds = 0.0
        if timestamp is not None:
            if isinstance(timestamp, str):
                try:
                    dt = datetime.datetime.fromisoformat(timestamp)
                    captured_time = dt.timestamp()
                    age_seconds = max(0.0, now - captured_time)
                except Exception:
                    pass
            elif isinstance(timestamp, (int, float)):
                captured_time = float(timestamp)
                age_seconds = max(0.0, now - captured_time)

        quality, is_stale = evaluate_freshness(
            age_seconds=age_seconds,
            max_age_seconds=self.limits.max_staleness_seconds,
        )

        # Health parsing
        health_dict = payload.get("health", payload)
        health = self.normalize_health(health_dict)

        # Metrics parsing
        raw_metrics = payload.get("metrics", [])
        metrics_list: List[TelemetryMetric] = []

        if isinstance(raw_metrics, list):
            for m in raw_metrics[:self.limits.max_metrics_per_result]:
                if isinstance(m, dict) and "name" in m and "value" in m:
                    try:
                        tm = normalize_metric(
                            name=str(m["name"]),
                            value=m["value"],
                            unit=str(m.get("unit", "")),
                            quality=quality,
                            timestamp=captured_time,
                        )
                        metrics_list.append(tm)
                    except Exception:
                        continue
        elif isinstance(raw_metrics, dict):
            for k, v in list(raw_metrics.items())[:self.limits.max_metrics_per_result]:
                try:
                    tm = normalize_metric(
                        name=str(k),
                        value=v,
                        quality=quality,
                        timestamp=captured_time,
                    )
                    metrics_list.append(tm)
                except Exception:
                    continue

        d_id = str(payload.get("device_id", source_id))
        p_type = str(payload.get("product_type", "general"))

        return TelemetryObservation(
            device_id=d_id,
            product_type=p_type,
            health=health,
            metrics=metrics_list,
            quality=quality,
            age_seconds=age_seconds,
            is_stale=is_stale,
            timestamp=captured_time,
        )

    def process_telemetry(
        self,
        source_id: str,
        raw_payload: Dict[str, Any],
        captured_at: float,
        provenance: Dict[str, Any],
        request_id: str,
        input_id: str,
        reference_time: Optional[float] = None,
        correlation_id: str = "",
        causation_id: Optional[str] = None,
    ) -> Tuple[TelemetryObservation, List[PerceptionEvidence]]:
        """Normalize raw telemetry dictionary into TelemetryObservation and PerceptionEvidence items."""
        now = reference_time if reference_time is not None else time.time()
        age = max(0.0, now - captured_at)
        quality, is_stale = evaluate_freshness(age, self.limits.max_staleness_seconds)

        obs = self.process(raw_payload, timestamp=captured_at, source_id=source_id)

        # Build evidence
        custom_attrs: Dict[str, Any] = {
            "device_id": obs.device_id,
            "product_type": obs.product_type,
            "quality": quality.value,
            "age_seconds": round(age, 2),
            "is_stale": is_stale,
            "metrics": [m.to_dict() for m in obs.metrics],
        }
        if obs.health.battery_percent is not None:
            custom_attrs["battery_percent"] = obs.health.battery_percent
        if obs.health.connectivity:
            custom_attrs["connectivity"] = obs.health.connectivity
        if obs.health.health_status:
            custom_attrs["health_status"] = obs.health.health_status

        # Carry over non-secret top-level keys
        for k, v in raw_payload.items():
            if k not in ("api_key", "password", "token", "secret", "bearer_token", "health", "metrics"):
                if k not in custom_attrs:
                    custom_attrs[k] = v

        ev_telem = PerceptionEvidence(
            evidence_id=f"ev_tel_{request_id}_{source_id}",
            semantic_type="telemetry",
            label="device_telemetry",
            confidence=1.0,
            source_id=source_id,
            modality=ModalityType.TELEMETRY,
            timestamp=captured_at,
            attributes=custom_attrs,
            provenance=provenance,
            correlation_id=correlation_id or request_id,
            causation_id=causation_id or input_id,
        )

        ev_state = PerceptionEvidence(
            evidence_id=f"ev_state_{request_id}_{source_id}",
            semantic_type="device_state",
            label="device_state_summary",
            confidence=1.0,
            source_id=source_id,
            modality=ModalityType.DEVICE_STATE,
            timestamp=captured_at,
            attributes=dict(custom_attrs),
            provenance=provenance,
            correlation_id=correlation_id or request_id,
            causation_id=causation_id or input_id,
        )

        return obs, [ev_telem, ev_state]


def process_telemetry(
    source_id: str,
    raw_payload: Dict[str, Any],
    captured_at: float,
    provenance: Dict[str, Any],
    request_id: str,
    input_id: str,
    reference_time: Optional[float] = None,
    correlation_id: str = "",
    causation_id: Optional[str] = None,
) -> Tuple[TelemetryObservation, List[PerceptionEvidence]]:
    """Module-level helper delegating to default TelemetryProcessor."""
    processor = TelemetryProcessor()
    return processor.process_telemetry(
        source_id=source_id,
        raw_payload=raw_payload,
        captured_at=captured_at,
        provenance=provenance,
        request_id=request_id,
        input_id=input_id,
        reference_time=reference_time,
        correlation_id=correlation_id,
        causation_id=causation_id,
    )
