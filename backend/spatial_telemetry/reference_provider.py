"""ATLAS Phase 6.5c — Deterministic Reference & Mock Spatial Telemetry Provider.

Provides predictable, repeatable synthetic spatial and telemetry outputs for:
- All 4 edge products (ATLAS Vision, ATLAS Glass, ATLAS Drone, ATLAS Rover)
- Digital Twin simulation scenario verification
- Unit and regression testing without hardware or network access
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.interfaces.perception_interface import PerceptionProviderInterface
from core.models.computer import ComputerObservation
from core.models.device_contract import (
    ConnectivityStatus,
    DeviceHealthStatus,
    ProductType,
    sanitize_contract_metadata,
)
from core.models.orchestration import GeoLocation, ModalityType
from core.models.perception import (
    PerceptionCapability,
    PerceptionError,
    PerceptionEvidence,
    PerceptionMetadata,
    PerceptionPrivacyClass,
    PerceptionRequest,
    PerceptionResult,
    PerceptionStatus,
    REQUIRED_PROVENANCE_KEYS,
    SpatialEvidence,
)
from core.models.spatial_telemetry import (
    PositionObservation,
    SpatialObservation,
    SpatialTelemetryLimits,
    TelemetryHealth,
    TelemetryMetric,
    TelemetryObservation,
    TelemetryQuality,
)


class CallableSet(set):
    def __call__(self) -> "CallableSet":
        return self


class MockSpatialTelemetryProvider(PerceptionProviderInterface):
    """
    Deterministic mock provider producing predictable spatial and telemetry evidence.
    Fully isolated from hardware, operating system GPS, and network telemetry feeds.
    """

    def __init__(
        self,
        provider_id: str = "mock_spatial_telemetry_provider",
        provider_version: str = "1.0.0",
        limits: Optional[SpatialTelemetryLimits] = None,
        available: bool = True,
        simulate_status: Optional[PerceptionStatus] = None,
        simulate_error: Optional[PerceptionError] = None,
        simulated_evidence: Optional[Union[List[PerceptionEvidence], Tuple[PerceptionEvidence, ...]]] = None,
    ) -> None:
        self._provider_id = provider_id
        self._provider_version = provider_version
        self.limits = limits or SpatialTelemetryLimits()
        self._available = available
        self.simulate_status = simulate_status
        self.simulate_error = simulate_error
        self._simulated_evidence = tuple(simulated_evidence) if simulated_evidence is not None else None
        self.call_count = 0

        self._supported_capabilities: Set[PerceptionCapability] = {
            PerceptionCapability.SPATIAL_LOCALIZATION,
            PerceptionCapability.TELEMETRY_NORMALIZATION,
        }

        self._supported_modalities: Set[ModalityType] = {
            ModalityType.GPS,
            ModalityType.TELEMETRY,
            ModalityType.DEVICE_STATE,
        }

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return self._provider_version

    @property
    def version(self) -> str:
        return self._provider_version

    @property
    def capabilities(self) -> CallableSet:
        return CallableSet(self._supported_capabilities)

    @property
    def supported_capabilities(self) -> CallableSet:
        return CallableSet(self._supported_capabilities)

    @property
    def supported_modalities(self) -> CallableSet:
        return CallableSet(self._supported_modalities)

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available

    def set_simulated_evidence(
        self, evidence: Optional[Union[List[PerceptionEvidence], Tuple[PerceptionEvidence, ...]]]
    ) -> None:
        self._simulated_evidence = tuple(evidence) if evidence is not None else None

    def get_health(self) -> Dict[str, Any]:
        return {
            "provider_id": self._provider_id,
            "provider_version": self._provider_version,
            "status": "HEALTHY" if self._available else "UNAVAILABLE",
            "available": self._available,
            "call_count": self.call_count,
            "supported_capabilities": [c.value for c in self._supported_capabilities],
            "supported_modalities": [m.value for m in self._supported_modalities],
        }

    def health_check(self) -> Any:
        class HealthResult:
            def __init__(self, d: Dict[str, Any]):
                self.status = d["status"]
                self.provider_id = d["provider_id"]
                self.error = None if d["available"] else "Provider unavailable"
                self.details = d

            def __bool__(self) -> bool:
                return self.status == "HEALTHY"

            def __eq__(self, other: Any) -> bool:
                if isinstance(other, bool):
                    return (self.status == "HEALTHY") == other
                return super().__eq__(other)

            def __getitem__(self, item: str) -> Any:
                return self.details[item]

        return HealthResult(self.get_health())

    def process(
        self, request_or_observation: Union[PerceptionRequest, ComputerObservation]
    ) -> Union[PerceptionResult, List[Any]]:
        return self.perceive(request_or_observation)

    def perceive(
        self,
        request_or_observation: Union[PerceptionRequest, ComputerObservation],
    ) -> Union[PerceptionResult, List[Any]]:
        self.call_count += 1

        if isinstance(request_or_observation, ComputerObservation):
            return []

        if not isinstance(request_or_observation, PerceptionRequest):
            raise TypeError(
                f"Unsupported request type for MockSpatialTelemetryProvider: {type(request_or_observation)}"
            )

        request = request_or_observation
        req_id = request.request_id
        inp_id = request.input_data.input_id
        source_id = request.input_data.source_id
        timestamp = request.input_data.captured_at

        meta = PerceptionMetadata(
            provider_id=self._provider_id,
            provider_version=self._provider_version,
            privacy_classification=request.privacy_constraints,
            processing_time_ms=1.0,
        )

        # 1. Simulated status / error override
        if self.simulate_status is not None:
            err = self.simulate_error or PerceptionError(
                code="SIMULATED_FAILURE",
                message="Mock simulated spatial telemetry error",
                provider_id=self._provider_id,
                request_id=req_id,
                input_id=inp_id,
                recoverable=True,
            )
            return PerceptionResult(
                request_id=req_id,
                input_id=inp_id,
                status=self.simulate_status,
                processing_metadata=meta,
                errors=(err,),
            )

        # 2. Availability check
        if not self._available:
            err = PerceptionError(
                code="PROVIDER_UNAVAILABLE",
                message="Mock spatial telemetry provider is unavailable",
                provider_id=self._provider_id,
                request_id=req_id,
                input_id=inp_id,
                recoverable=True,
            )
            return PerceptionResult(
                request_id=req_id,
                input_id=inp_id,
                status=PerceptionStatus.FAILED,
                processing_metadata=meta,
                errors=(err,),
            )

        # 3. Modality check
        if request.input_data.modality not in self._supported_modalities:
            err = PerceptionError(
                code="UNSUPPORTED_MODALITY",
                message=f"Modality '{request.input_data.modality.value}' not supported by mock provider",
                provider_id=self._provider_id,
                request_id=req_id,
                input_id=inp_id,
                recoverable=False,
            )
            return PerceptionResult(
                request_id=req_id,
                input_id=inp_id,
                status=PerceptionStatus.UNSUPPORTED_MODALITY,
                processing_metadata=meta,
                errors=(err,),
            )

        # 4. Custom simulated evidence override
        if self._simulated_evidence is not None:
            return PerceptionResult(
                request_id=req_id,
                input_id=inp_id,
                status=PerceptionStatus.SUCCESS,
                evidence=self._simulated_evidence,
                processing_metadata=meta,
            )

        # 5. Deterministic synthetic outputs matching the product role
        product_role = request.input_data.metadata.get("product_role", "drone")
        if "vision" in source_id.lower() or product_role == "surveillance":
            product_type = ProductType.VISION
        elif "glass" in source_id.lower() or product_role == "wearable_hud":
            product_type = ProductType.GLASS
        elif "rover" in source_id.lower() or product_role == "ground_patrol":
            product_type = ProductType.ROVER
        else:
            product_type = ProductType.DRONE

        provenance = {
            "source_id": source_id,
            "provider_id": self._provider_id,
            "provider_version": self._provider_version,
            "input_id": inp_id,
            "request_id": req_id,
            "observation_timestamp": timestamp,
            "correlation_id": request.correlation_id or req_id,
            "causation_id": request.causation_id or inp_id,
            "device_id": source_id,
            "product_type": product_type.value,
        }

        evidence_list: List[PerceptionEvidence] = []

        if product_type == ProductType.VISION:
            # Fixed surveillance hub
            geo = GeoLocation(latitude=37.7833, longitude=-122.4167, altitude=15.0, accuracy=2.0)
            spatial_ev = SpatialEvidence(point=(37.7833, -122.4167), location=geo, spatial_confidence=1.0)
            ev_pos = PerceptionEvidence(
                evidence_id=f"ev_pos_{req_id}_vision",
                semantic_type="spatial_position",
                label="fixed_camera_location",
                confidence=1.0,
                source_id=source_id,
                modality=ModalityType.GPS,
                timestamp=timestamp,
                spatial=spatial_ev,
                attributes={"latitude": 37.7833, "longitude": -122.4167, "altitude": 15.0, "is_fixed": True},
                provenance=provenance,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )
            ev_tel = PerceptionEvidence(
                evidence_id=f"ev_tel_{req_id}_vision",
                semantic_type="telemetry",
                label="device_telemetry",
                confidence=1.0,
                source_id=source_id,
                modality=ModalityType.TELEMETRY,
                timestamp=timestamp,
                attributes={
                    "battery_pct": 100.0,
                    "connectivity": ConnectivityStatus.ONLINE.value,
                    "health_status": DeviceHealthStatus.HEALTHY.value,
                    "temperature_celsius": 32.5,
                },
                provenance=provenance,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )
            evidence_list.extend([ev_pos, ev_tel])

        elif product_type == ProductType.GLASS:
            # Wearable HUD
            geo = GeoLocation(latitude=37.7755, longitude=-122.4180, altitude=2.0, accuracy=3.5)
            spatial_ev = SpatialEvidence(point=(37.7755, -122.4180), location=geo, spatial_confidence=0.95)
            ev_pos = PerceptionEvidence(
                evidence_id=f"ev_pos_{req_id}_glass",
                semantic_type="spatial_position",
                label="wearer_gps_location",
                confidence=0.95,
                source_id=source_id,
                modality=ModalityType.GPS,
                timestamp=timestamp,
                spatial=spatial_ev,
                attributes={"latitude": 37.7755, "longitude": -122.4180, "speed": 1.4, "heading": 45.0},
                provenance=provenance,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )
            ev_tel = PerceptionEvidence(
                evidence_id=f"ev_tel_{req_id}_glass",
                semantic_type="telemetry",
                label="device_telemetry",
                confidence=1.0,
                source_id=source_id,
                modality=ModalityType.TELEMETRY,
                timestamp=timestamp,
                attributes={
                    "battery_pct": 88.0,
                    "connectivity": ConnectivityStatus.ONLINE.value,
                    "health_status": DeviceHealthStatus.HEALTHY.value,
                    "temperature_celsius": 28.0,
                },
                provenance=provenance,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )
            evidence_list.extend([ev_pos, ev_tel])

        elif product_type == ProductType.ROVER:
            # Ground Rover
            geo = GeoLocation(latitude=37.7740, longitude=-122.4200, altitude=10.0, accuracy=1.5)
            spatial_ev = SpatialEvidence(point=(37.7740, -122.4200), location=geo, spatial_confidence=0.98)
            ev_pos = PerceptionEvidence(
                evidence_id=f"ev_pos_{req_id}_rover",
                semantic_type="spatial_position",
                label="rover_gps_location",
                confidence=0.98,
                source_id=source_id,
                modality=ModalityType.GPS,
                timestamp=timestamp,
                spatial=spatial_ev,
                attributes={"latitude": 37.7740, "longitude": -122.4200, "speed": 1.2, "heading": 180.0},
                provenance=provenance,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )
            ev_tel = PerceptionEvidence(
                evidence_id=f"ev_tel_{req_id}_rover",
                semantic_type="telemetry",
                label="device_telemetry",
                confidence=1.0,
                source_id=source_id,
                modality=ModalityType.TELEMETRY,
                timestamp=timestamp,
                attributes={
                    "battery_pct": 34.0,
                    "connectivity": ConnectivityStatus.DEGRADED.value,
                    "health_status": DeviceHealthStatus.HEALTHY.value,
                    "temperature_celsius": 38.0,
                },
                provenance=provenance,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )
            evidence_list.extend([ev_pos, ev_tel])

        else:
            # Aerial Drone
            geo = GeoLocation(latitude=37.7749, longitude=-122.4194, altitude=45.0, accuracy=0.8)
            spatial_ev = SpatialEvidence(point=(37.7749, -122.4194), location=geo, spatial_confidence=0.99)
            ev_pos = PerceptionEvidence(
                evidence_id=f"ev_pos_{req_id}_drone",
                semantic_type="spatial_position",
                label="drone_flight_position",
                confidence=0.99,
                source_id=source_id,
                modality=ModalityType.GPS,
                timestamp=timestamp,
                spatial=spatial_ev,
                attributes={"latitude": 37.7749, "longitude": -122.4194, "altitude": 45.0, "speed": 8.5, "heading": 90.0},
                provenance=provenance,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )
            ev_tel = PerceptionEvidence(
                evidence_id=f"ev_tel_{req_id}_drone",
                semantic_type="telemetry",
                label="device_telemetry",
                confidence=1.0,
                source_id=source_id,
                modality=ModalityType.TELEMETRY,
                timestamp=timestamp,
                attributes={
                    "battery_pct": 71.0,
                    "connectivity": ConnectivityStatus.ONLINE.value,
                    "health_status": DeviceHealthStatus.HEALTHY.value,
                    "temperature_celsius": 35.0,
                    "metric_speed": 8.5,
                    "metric_altitude": 45.0,
                },
                provenance=provenance,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )
            evidence_list.extend([ev_pos, ev_tel])

        # Filter by requested modality if specific
        if request.input_data.modality == ModalityType.GPS:
            evidence_list = [e for e in evidence_list if e.modality == ModalityType.GPS]
        elif request.input_data.modality == ModalityType.TELEMETRY:
            evidence_list = [e for e in evidence_list if e.modality == ModalityType.TELEMETRY]
        elif request.input_data.modality == ModalityType.DEVICE_STATE:
            evidence_list = [e for e in evidence_list if e.modality == ModalityType.DEVICE_STATE]

        return PerceptionResult(
            request_id=req_id,
            input_id=inp_id,
            status=PerceptionStatus.SUCCESS,
            evidence=tuple(evidence_list),
            processing_metadata=meta,
            created_at=time.time(),
        )


ReferenceSpatialTelemetryProvider = MockSpatialTelemetryProvider
