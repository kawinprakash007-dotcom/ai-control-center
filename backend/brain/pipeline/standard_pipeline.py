from typing import Any, Optional, List, Dict

from core.interfaces.pipeline_interface import PipelineInterface
from core.interfaces.request_understanding_interface import RequestUnderstandingInterface
from core.interfaces.decision_engine_interface import DecisionEngineInterface
from core.interfaces.decision_planner_interface import DecisionPlannerInterface
from core.interfaces.execution_engine_interface import ExecutionEngineInterface
from core.interfaces.verification_interface import VerificationInterface
from core.interfaces.response_composer_interface import ResponseComposerInterface
from core.interfaces.memory_interface import MemoryServiceInterface
from core.interfaces.web_interface import WebProviderInterface
from core.models.pipeline import PipelineResult
from core.models.memory import MessageRole

from brain.request_understanding import StandardRequestUnderstanding
from brain.decision_engine import StandardDecisionEngine
from brain.planning import StandardPlanner
from brain.execution import StandardExecutionEngine
from brain.verification import StandardVerifier
from brain.response import StandardResponseComposer
from memory.sqlite_store import SQLiteMemoryStore


class StandardPipeline(PipelineInterface):
    """
    Deterministic implementation of PipelineInterface.
    Orchestrates the Phase 2 request lifecycle:
    Understanding -> Memory Read -> Decision -> Planning -> Execution -> Verification -> Response -> Memory Write.
    """

    def __init__(
        self,
        understanding: Optional[RequestUnderstandingInterface] = None,
        decision_engine: Optional[DecisionEngineInterface] = None,
        planner: Optional[DecisionPlannerInterface] = None,
        execution_engine: Optional[ExecutionEngineInterface] = None,
        verifier: Optional[VerificationInterface] = None,
        composer: Optional[ResponseComposerInterface] = None,
        memory_service: Optional[MemoryServiceInterface] = None,
        web_provider: Optional[WebProviderInterface] = None,
    ):
        self.understanding = (
            understanding
            if understanding is not None
            else StandardRequestUnderstanding()
        )
        self.decision_engine = (
            decision_engine
            if decision_engine is not None
            else StandardDecisionEngine()
        )
        self.planner = (
            planner
            if planner is not None
            else StandardPlanner()
        )
        self.execution_engine = (
            execution_engine
            if execution_engine is not None
            else StandardExecutionEngine()
        )
        self.verifier = (
            verifier
            if verifier is not None
            else StandardVerifier()
        )
        self.composer = (
            composer
            if composer is not None
            else StandardResponseComposer()
        )
        self.memory_service = (
            memory_service
            if memory_service is not None
            else SQLiteMemoryStore()
        )
        self.web_provider = web_provider

    def process(self, input_data: Any) -> PipelineResult:
        """
        Execute the full Phase 2 lifecycle and return structured PipelineResult.
        """
        request = self.understanding.understand(input_data)

        # Memory Read: retrieve prior conversational history for this session
        is_empty = bool(request.parameters.get("is_empty", False))
        history_entries = []
        if not is_empty:
            history_entries = self.memory_service.get_history(request.session_id, limit=20)

        decision = self.decision_engine.decide(request)
        plan = self.planner.plan(decision)

        # Supply conversation context to tasks without mutating Request
        if plan.steps and not is_empty:
            formatted_history: List[Dict[str, str]] = [
                {"role": msg.role.value, "content": msg.content}
                for msg in history_entries
            ]
            for step in plan.steps:
                if getattr(step, "type", None) == "chat" or getattr(step, "tool", None) == "chat":
                    params = dict(step.parameters) if step.parameters else {}
                    if "query" not in params:
                        params["query"] = request.original_text
                    if "history" not in params:
                        params["history"] = formatted_history
                    step.parameters = params
                elif getattr(step, "type", None) == "memory" or getattr(step, "tool", None) == "memory":
                    params = dict(step.parameters) if step.parameters else {}
                    if "memory_service" not in params and self.memory_service is not None:
                        params["memory_service"] = self.memory_service
                    if "user_id" not in params:
                        params["user_id"] = "default_user"
                    step.parameters = params
                elif getattr(step, "type", None) == "web" or getattr(step, "tool", None) == "web":
                    params = dict(step.parameters) if step.parameters else {}
                    if "web_provider" not in params and self.web_provider is not None:
                        params["web_provider"] = self.web_provider
                    step.parameters = params



        results = self.execution_engine.execute(plan)
        verification = self.verifier.verify(plan, results)
        response = self.composer.compose(
            request,
            decision,
            plan,
            results,
            verification,
        )

        # Memory Write: persist user turn and assistant response exactly once
        if not is_empty:
            self.memory_service.add_message(
                session_id=request.session_id,
                role=MessageRole.USER,
                content=request.original_text,
            )
            self.memory_service.add_message(
                session_id=request.session_id,
                role=MessageRole.ASSISTANT,
                content=response,
            )

        return PipelineResult(
            response=response,
            request=request,
            decision=decision,
            plan=plan,
            results=results,
            verification=verification,
        )

    def run(self, input_data: Any) -> str:
        """
        Execute the full Phase 2 lifecycle and return only the final response string.
        Delegates directly to process().
        """
        return self.process(input_data).response
