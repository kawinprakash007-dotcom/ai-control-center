import asyncio
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, status
from pydantic import BaseModel, Field

from core.security import verify_api_token, verify_ws_token
from core.models.orchestration import MultimodalObservation, ModalityType, GeoLocation
from orchestration.input_gateway import IngressEnvelope
from core.models.tool_call import ToolCall

router = APIRouter(prefix="/api/v1", tags=["ATLAS Central API"])


# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------

class ObservationIngressRequest(BaseModel):
    observation_id: str
    source_id: str
    source_type: str
    modality: str
    payload: Any
    timestamp: Optional[float] = Field(default_factory=time.time)
    confidence: float = 1.0
    location: Optional[Dict[str, Any]] = None
    device_id: Optional[str] = None
    correlation_id: Optional[str] = ""
    causation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"
    user_id: Optional[str] = "default_user"


class DeviceCommandRequest(BaseModel):
    capability: str
    action: str
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    correlation_id: Optional[str] = ""


# ---------------------------------------------------------------------------
# Health & Readiness Endpoints (Public)
# ---------------------------------------------------------------------------

@router.get("/health")
def get_health(request: Request):
    atlas = request.app.state.atlas
    return {
        "status": "healthy" if not atlas.shutting_down else "shutting_down",
        "timestamp": time.time(),
        "app_env": atlas.settings.app_env,
        "demo_mode": atlas.settings.demo_mode,
    }


@router.get("/ready")
def get_ready(request: Request):
    atlas = request.app.state.atlas
    if not atlas.ready or atlas.shutting_down:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ATLAS Central Runtime is initializing or shutting down",
        )
    return {
        "status": "ready",
        "components": {
            "central_orchestrator": atlas.central_orchestrator is not None,
            "device_gateway": atlas.device_gateway is not None,
            "input_gateway": atlas.input_gateway is not None,
            "fusion_engine": atlas.fusion_engine is not None,
            "world_store": atlas.world_store is not None,
            "goal_manager": atlas.goal_manager is not None,
            "cognitive_runtime": atlas.cognitive_runtime is not None,
        },
        "simulation_mode": atlas.settings.is_simulation(),
        "demo_mode": atlas.settings.demo_mode,
    }


# ---------------------------------------------------------------------------
# Protected Endpoints (Token Verified)
# ---------------------------------------------------------------------------

@router.post("/chat")
async def post_chat(
    data: ChatRequest,
    request: Request,
    token: str = Depends(verify_api_token),
):
    """
    Modern chat endpoint executing turns through CognitiveRuntime.
    Offloaded to a bounded worker thread to ensure the event loop remains responsive.
    """
    atlas = request.app.state.atlas
    if not atlas.ready:
        raise HTTPException(status_code=503, detail="Runtime not ready")

    loop = asyncio.get_running_loop()
    try:
        turn_result = await loop.run_in_executor(
            atlas.executor,
            atlas.cognitive_runtime.execute_turn,
            data.message,
        )
        if atlas.trace_store and turn_result.trace:
            try:
                atlas.trace_store.save_trace(turn_result.trace)
            except Exception:
                pass
        execution_time = turn_result.trace.duration_seconds if (turn_result.trace and hasattr(turn_result.trace, "duration_seconds")) else 0.0
        trace_id = turn_result.trace.trace_id if (turn_result.trace and hasattr(turn_result.trace, "trace_id")) else ""
        return {
            "turn_id": turn_result.turn_id,
            "status": turn_result.status.value if hasattr(turn_result.status, "value") else str(turn_result.status),
            "response": turn_result.response,
            "execution_time": execution_time,
            "trace_id": trace_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cognitive turn failed: {str(e)}")


@router.post("/ingress/observation")
def post_observation(
    data: ObservationIngressRequest,
    request: Request,
    token: str = Depends(verify_api_token),
):
    """
    Ingress boundary endpoint.
    Converts request to MultimodalObservation and routes through CentralOrchestrator.
    """
    atlas = request.app.state.atlas
    if not atlas.ready:
        raise HTTPException(status_code=503, detail="Runtime not ready")

    try:
        loc = GeoLocation.from_dict(data.location) if data.location else None
        obs = MultimodalObservation(
            observation_id=data.observation_id,
            source_id=data.source_id,
            source_type=data.source_type,
            modality=ModalityType.from_str(data.modality),
            timestamp=data.timestamp or time.time(),
            payload=data.payload,
            confidence=data.confidence,
            location=loc,
            device_id=data.device_id,
            correlation_id=data.correlation_id or data.observation_id,
            causation_id=data.causation_id,
            metadata=data.metadata or {},
        )
        env = IngressEnvelope(
            message_id=obs.observation_id,
            source_id=obs.source_id,
            source_type=obs.source_type,
            modality=obs.modality,
            timestamp=obs.timestamp,
            payload=obs.payload,
            confidence=obs.confidence,
            location=obs.location,
            device_id=obs.device_id,
            correlation_id=obs.correlation_id,
            causation_id=obs.causation_id,
            metadata=obs.metadata,
        )
        res = atlas.central_orchestrator.process_ingress(env)
        is_success = res.status in ("SUCCESS", "NOOP", "NO_ACTION")

        # Extract structured details for judge-facing traceability
        serialized_situations = [
            {
                "situation_id": s.situation_id,
                "category": s.category.value if hasattr(s.category, "value") else str(s.category),
                "severity": s.severity.value if hasattr(s.severity, "value") else str(s.severity),
                "status": s.status.value if hasattr(s.status, "value") else str(s.status),
                "title": getattr(s, "title", ""),
                "description": getattr(s, "description", ""),
            }
            for s in res.situations_fused
        ]
        serialized_transitions = [
            {
                "transition_id": t.transition_id,
                "entity_id": t.entity_id,
                "property_name": t.property_name,
                "transition_type": t.transition_type.value if hasattr(t.transition_type, "value") else str(t.transition_type),
                "from_version": t.from_version,
                "to_version": t.to_version,
            }
            for t in res.world_transitions
        ]
        serialized_goals = [
            {
                "goal_id": getattr(g, "id", getattr(g, "goal_id", "")),
                "title": getattr(g, "title", getattr(g, "original_goal", "")),
                "status": g.status.value if hasattr(g.status, "value") else str(g.status),
                "priority": g.priority.value if hasattr(g.priority, "value") else str(g.priority),
            }
            for g in res.goals_created
        ]
        serialized_tool_results = [
            {
                "capability": getattr(t, "capability", ""),
                "action": getattr(t, "action", ""),
                "success": getattr(t, "success", True),
                "call_id": getattr(t, "call_id", ""),
            }
            for t in res.tool_results
        ]

        return {
            "success": is_success,
            "status": res.status,
            "observation_id": obs.observation_id,
            "correlation_id": obs.correlation_id,
            "orchestration_id": res.cycle_id,
            "cycle_id": res.cycle_id,
            "cycle_count": 1 if res.cycle_id else 0,
            "observations_ingested_count": len(res.observations_ingested),
            "situations_fused_count": len(res.situations_fused),
            "world_transitions_count": len(res.world_transitions),
            "events_evaluated_count": len(res.events_evaluated),
            "autonomy_decisions_count": len(res.autonomy_decisions),
            "goals_created_count": len(res.goals_created),
            "tool_results_count": len(res.tool_results),
            "duration_seconds": res.duration_seconds,
            "causal_trace": res.causal_trace,
            "situations": serialized_situations,
            "world_transitions": serialized_transitions,
            "goals": serialized_goals,
            "tool_results": serialized_tool_results,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Observation validation error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration processing error: {str(e)}")


@router.get("/devices")
def list_devices(
    request: Request,
    token: str = Depends(verify_api_token),
):
    atlas = request.app.state.atlas
    if not atlas.ready:
        raise HTTPException(status_code=503, detail="Runtime not ready")
    return {"devices": atlas.device_gateway.list_devices()}


@router.get("/devices/{device_id}")
def get_device(
    device_id: str,
    request: Request,
    token: str = Depends(verify_api_token),
):
    atlas = request.app.state.atlas
    if not atlas.ready:
        raise HTTPException(status_code=503, detail="Runtime not ready")
    ident = atlas.device_gateway.get_device(device_id)
    if not ident:
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found")
    status_data = atlas.device_gateway.query_device_status(device_id)
    return {
        "device": ident.to_dict(),
        "status": status_data,
    }


@router.post("/devices/{device_id}/command")
def dispatch_device_command(
    device_id: str,
    data: DeviceCommandRequest,
    request: Request,
    token: str = Depends(verify_api_token),
):
    """
    Manual device dispatch endpoint.
    Routes strictly through PolicyEngine -> ToolOrchestrator -> DeviceGateway.
    """
    atlas = request.app.state.atlas
    if not atlas.ready:
        raise HTTPException(status_code=503, detail="Runtime not ready")

    tc = ToolCall(
        capability="device_gateway",
        action="dispatch_capability",
        parameters={
            "device_id": device_id,
            "capability": data.capability,
            "action": data.action,
            "parameters": data.parameters or {},
            "correlation_id": data.correlation_id or "",
        },
        call_id=f"api_call_{int(time.time()*1000)}",
    )
    result = atlas.tool_orchestrator.execute(tc)
    output_val = result.output if result.output is not None else (result.data if hasattr(result, "data") else None)
    if output_val is None:
        output_val = result.message
    return {
        "success": result.success,
        "message": result.message,
        "output": output_val,
        "data": getattr(result, "data", None),
        "error": getattr(result, "error", None) or (None if result.success else result.message),
    }


@router.get("/world/state")
def get_world_state(
    request: Request,
    token: str = Depends(verify_api_token),
):
    atlas = request.app.state.atlas
    if not atlas.ready:
        raise HTTPException(status_code=503, detail="Runtime not ready")
    current_state = atlas.world_store.get_current_state()
    return current_state.to_dict()


@router.get("/goals")
def get_goals(
    request: Request,
    token: str = Depends(verify_api_token),
):
    atlas = request.app.state.atlas
    if not atlas.ready:
        raise HTTPException(status_code=503, detail="Runtime not ready")
    goals = atlas.goal_store.get_all_goals() if hasattr(atlas.goal_store, "get_all_goals") else []
    return {"goals": [g.to_dict() for g in goals]}


@router.get("/traces")
def list_traces(
    limit: int = 50,
    request: Request = None,
    token: str = Depends(verify_api_token),
):
    atlas = request.app.state.atlas
    if not atlas.ready or not atlas.trace_store:
        raise HTTPException(status_code=503, detail="Runtime not ready or trace store unavailable")
    return {"traces": atlas.trace_store.list_traces(limit=limit)}


# ---------------------------------------------------------------------------
# WebSockets: Real-Time Telemetry & Glass HUD Channels
# ---------------------------------------------------------------------------

@router.websocket("/telemetry")
async def websocket_telemetry(websocket: WebSocket, token: Optional[str] = None):
    if not verify_ws_token(websocket, token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized: invalid token")
        return

    atlas = websocket.app.state.atlas
    accepted = await atlas.telemetry_manager.connect(websocket)
    if not accepted:
        return

    try:
        while True:
            # Keep-alive loop, discard client-sent messages or handle client pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except Exception:
        pass
    finally:
        atlas.telemetry_manager.disconnect(websocket)


@router.websocket("/glass/hud")
async def websocket_glass_hud(websocket: WebSocket, token: Optional[str] = None):
    if not verify_ws_token(websocket, token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized: invalid token")
        return

    atlas = websocket.app.state.atlas
    accepted = await atlas.hud_manager.connect(websocket)
    if not accepted:
        return

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except Exception:
        pass
    finally:
        atlas.hud_manager.disconnect(websocket)
