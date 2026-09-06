from typing import Optional, List, Dict, Any
import uuid

from core.interfaces.reasoning_interface import ReasoningProviderInterface
from core.models.reasoning import (
    ReasoningRequest,
    ReasoningResponse,
    ReasoningOutcome,
    ActionProposal,
    ProviderMetadata,
    ModelCapabilityType,
)


class OllamaReasoningProvider(ReasoningProviderInterface):
    """
    Optional adapter wrapping local Ollama client into the model-neutral
    ReasoningProviderInterface.
    Demonstrates model-neutral seam without coupling ATLAS to Ollama.
    """

    def __init__(
        self,
        provider_id: str = "local_ollama",
        model_id: str = "qwen3:8b",
        is_local: bool = True,
    ):
        self._provider_id = provider_id
        self._model_id = model_id
        self._metadata = ProviderMetadata(
            provider_id=self._provider_id,
            version="1.0.0",
            supported_capabilities=(
                ModelCapabilityType.TEXT,
                ModelCapabilityType.STRUCTURED_OUTPUT,
            ),
            is_local=is_local,
            description=f"Local Ollama provider running model {model_id}.",
            metadata={"model_id": self._model_id},
        )

    @property
    def metadata(self) -> ProviderMetadata:
        return self._metadata

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        """
        Invoke Ollama and wrap response into structured ReasoningResponse.
        """
        from llm.ollama_client import ask_ollama

        # Format history
        history_list = [{"role": h.get("role", "user"), "content": h.get("content", "")} for h in request.history]

        user_prompt = f"Goal: {request.goal}\nTask: {request.task_description or ''}"

        try:
            content = ask_ollama(user_prompt, history=history_list if history_list else None)
            return ReasoningResponse(
                turn_id=f"ollama_{uuid.uuid4().hex[:8]}",
                outcome=ReasoningOutcome.REPORT_COMPLETION,
                completion_summary=content,
                raw_output=content,
            )
        except Exception as e:
            return ReasoningResponse(
                turn_id=f"ollama_err_{uuid.uuid4().hex[:8]}",
                outcome=ReasoningOutcome.ABORT,
                abort_reason=f"Ollama execution failed: {str(e)}",
                confidence=0.0,
            )
