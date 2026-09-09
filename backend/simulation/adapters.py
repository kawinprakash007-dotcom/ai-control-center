"""
ATLAS Phase 6.3 — Digital Twin Adapter Boundary.

Provides the transparent protocol adapter connecting digital twins to DeviceGateway.
CRITICAL ARCHITECTURAL GUARANTEE:
A future physical adapter and a digital twin adapter satisfy the EXACT SAME
DeviceAdapterInterface and DeviceContract models.
Central Orchestrator, DeviceGateway, and PolicyEngine operate completely
unaware whether they are communicating with a digital twin or physical hardware.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.interfaces.orchestration_interface import DeviceAdapterInterface
from core.interfaces.simulation_interface import DigitalTwinInterface
from core.models.device_contract import (
    DeviceHealth,
    DeviceHeartbeat,
    ProductRole,
    ProductType,
)
from core.models.orchestration import (
    ConnectivityStatus,
    DeviceCapabilityDescriptor,
    DeviceIdentity,
    DeviceType,
)
from core.models.result import Result
from core.models.tool_call import ToolCall
from orchestration.device_gateway import DeviceCommand


class DigitalTwinAdapter(DeviceAdapterInterface):
    """
    Adapter implementing DeviceAdapterInterface by delegating to a DigitalTwinInterface.
    Ensures 100% contract equivalence between digital twins and future physical adapters.
    """

    def __init__(self, twin: DigitalTwinInterface):
        self.twin = twin

    def get_protocol_name(self) -> str:
        return "digital_twin"

    def connect(self, device_id: str = "") -> bool:
        return True

    def disconnect(self, device_id: str = "") -> bool:
        return True

    def get_status(self, device_id: str = "") -> ConnectivityStatus:
        state = self.twin.get_state()
        return state.connectivity

    def get_health(self, device_id: str = "") -> DeviceHealth:
        state = self.twin.get_state()
        telem = self.twin.get_telemetry()
        cap_dict = {cap: True for cap in state.active_capabilities}
        return DeviceHealth(
            device_id=self.twin.twin_id,
            status=state.health,
            connectivity=state.connectivity,
            battery_pct=state.battery,
            capability_availability=cap_dict,
            last_heartbeat=state.last_update,
        )

    def heartbeat(self, device_id: str = "") -> DeviceHeartbeat:
        state = self.twin.get_state()
        return DeviceHeartbeat(
            device_id=self.twin.twin_id,
            timestamp=state.last_update,
            connectivity_state=state.connectivity,
            health_summary=state.health,
            metrics={"simulated": True, "battery": state.battery},
        )

    def get_capabilities(self, device_id: str = "") -> Sequence[DeviceCapabilityDescriptor]:
        state = self.twin.get_state()
        descriptors = []
        for cap_id in state.active_capabilities:
            descriptors.append(
                DeviceCapabilityDescriptor(
                    capability_name=cap_id,
                    action_name=cap_id,
                    capability_id=cap_id,
                    description=f"Simulated capability {cap_id} for {self.twin.twin_id}",
                    supported_actions=(cap_id,),
                    schema_version=state.schema_version,
                )
            )
        return descriptors

    def format_command(self, tool_call: ToolCall) -> Dict[str, Any]:
        return {
            "protocol": self.get_protocol_name(),
            "action": tool_call.action,
            "params": tool_call.parameters,
        }

    def execute_command(self, command: DeviceCommand) -> Result:
        """Forward command execution directly to the digital twin."""
        return self.twin.execute_command(command)


def create_digital_twin_device(
    twin: DigitalTwinInterface,
) -> Tuple[DeviceIdentity, DigitalTwinAdapter]:
    """
    Factory creating a canonical DeviceIdentity and DigitalTwinAdapter
    ready for registration into DeviceGateway.
    """
    state = twin.get_state()
    adapter = DigitalTwinAdapter(twin)
    capabilities = adapter.get_capabilities()
    identity = DeviceIdentity(
        device_id=twin.twin_id,
        device_type=DeviceType.from_str(state.product_type.value),
        display_name=f"{state.product_type.value} Twin {twin.twin_id}",
        product_type=state.product_type,
        product_role=state.product_role,
        capabilities=tuple(capabilities),
        connectivity_status=state.connectivity,
        is_simulation=True,
        metadata={"simulation": True, "twin_id": twin.twin_id},
    )
    return identity, adapter
