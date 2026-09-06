import os
from brain.agent import Agent
from brain.pipeline import StandardPipeline
from brain.request_understanding import RequestUnderstandingError

# V1 Brain instance
jarvis = Agent()

# Phase 2 Pipeline instance
pipeline = StandardPipeline()


def is_phase2_enabled() -> bool:
    """Check whether the Phase 2 brain pipeline is enabled via environment flag."""
    val = os.getenv("USE_PHASE2_BRAIN", "false").lower().strip()
    return val in ("true", "1", "yes", "on")


def process_message(message: str) -> str:
    """
    Main entry point for processing incoming user messages.
    Dispatches to Phase 2 pipeline when USE_PHASE2_BRAIN=True with safe
    pre-execution fallback to V1, or directly to legacy V1 Agent when False.
    """
    if not is_phase2_enabled():
        return jarvis.think(message)

    # Phase 2 Primary Path
    try:
        result = pipeline.process(message)
        return result.response

    except RequestUnderstandingError:
        raise

    except ValueError:
        # Pre-execution planning failure (e.g. unsupported capability in Phase 2 planner).
        # Safe to fall back to V1 because Phase 2 execution never started and memory was not written.
        return jarvis.think(message)
