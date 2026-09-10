import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import AtlasSettings, get_settings
from core.app_state import initialize_application_state, shutdown_application_state
from core.api_routes import router as api_v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager.
    Initializes ATLAS application state on startup and cleans up on shutdown.
    """
    settings = getattr(app.state, "settings", None) or get_settings()
    in_memory = getattr(app.state, "in_memory_stores", False)

    # Initialize complete ATLAS Central Runtime
    atlas_state = initialize_application_state(settings=settings, in_memory_stores=in_memory)
    app.state.atlas = atlas_state

    yield

    # Clean shutdown
    if hasattr(app.state, "atlas") and app.state.atlas:
        await shutdown_application_state(app.state.atlas)


def create_app(settings: Optional[AtlasSettings] = None, in_memory_stores: bool = False) -> FastAPI:
    """
    Application factory for the ATLAS Central Orchestration Platform.
    """
    cfg = settings or get_settings()

    app = FastAPI(
        title="ATLAS AI Control Center",
        description="Production Central Orchestration and Real-Time Ingress Platform",
        version="6.1.0",
        lifespan=lifespan,
    )

    app.state.settings = cfg
    app.state.in_memory_stores = in_memory_stores

    # Restricted CORS configuration (no unrestricted wildcard with credentials)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register versioned API routes
    app.include_router(api_v1_router)

    # Backward compatibility root endpoints
    @app.get("/")
    def home():
        return {"status": "running", "platform": "ATLAS Central Orchestrator", "version": "6.1.0"}

    @app.get("/health")
    def root_health():
        return {"status": "healthy", "platform": "ATLAS Central Orchestrator", "demo_mode": getattr(cfg, "demo_mode", False)}

    @app.get("/ready")
    def root_ready(request: Request):
        atlas = getattr(request.app.state, "atlas", None)
        ready = atlas is not None and getattr(atlas, "ready", False)
        return {"status": "ready" if ready else "initializing", "simulation_mode": getattr(cfg, "simulation_mode", True), "demo_mode": getattr(cfg, "demo_mode", False)}

    # Backward compatibility for legacy frontend client calling /chat
    class LegacyChatRequest(BaseModel):
        message: str

    @app.post("/chat")
    async def legacy_chat(data: LegacyChatRequest, request: Request):
        """
        Backward compatibility wrapper.
        Routes incoming chats directly through the modern CognitiveRuntime
        without touching legacy V1 Agent.
        """
        atlas = getattr(request.app.state, "atlas", None)
        if atlas and getattr(atlas, "ready", False):
            loop = asyncio.get_running_loop()
            turn_result = await loop.run_in_executor(
                atlas.executor,
                atlas.cognitive_runtime.execute_turn,
                data.message,
            )
            return {
                "response": turn_result.response,
                "turn_id": turn_result.turn_id,
            }
        else:
            # Fallback for unmanaged test harnesses where lifespan was not invoked
            from brain.assistant import process_message
            return {"response": process_message(data.message)}

    return app


# Default ASGI application instance for Uvicorn
app = create_app()