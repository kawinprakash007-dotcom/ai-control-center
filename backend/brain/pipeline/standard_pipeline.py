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
from core.interfaces.recovery_interface import RecoveryEngineInterface
from core.models.pipeline import PipelineResult
from core.models.memory import MessageRole
from core.models.request import Request
from core.models.plan import Plan
from core.models.result import Result

from brain.request_understanding import StandardRequestUnderstanding
from brain.decision_engine import StandardDecisionEngine
from brain.planning import StandardPlanner
from brain.execution import StandardExecutionEngine
from brain.verification import StandardVerifier
from brain.response import StandardResponseComposer
from memory.sqlite_store import SQLiteMemoryStore
from runtime.cognitive_runtime import CognitiveRuntime


class StandardPipeline(PipelineInterface):
    """
    Deterministic implementation of PipelineInterface.
    Orchestrates the Phase 2 & Phase 3 request lifecycle backed by CognitiveRuntime:
    Understanding -> Memory Read -> Decision -> Planning -> Execution -> Verification -> (Recovery) -> Response -> Memory Write.
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
        recovery_engine: Optional[RecoveryEngineInterface] = None,
        enable_recovery: bool = True,
        runtime: Optional[CognitiveRuntime] = None,
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

        if recovery_engine is not None:
            self.recovery_engine = recovery_engine
        elif enable_recovery:
            try:
                from brain.recovery.recovery_engine import StandardRecoveryEngine
                self.recovery_engine = StandardRecoveryEngine()
            except ImportError:
                self.recovery_engine = None
        else:
            self.recovery_engine = None

        if runtime is not None:
            self.runtime = runtime
        else:
            self.runtime = CognitiveRuntime(
                understanding=self.understanding,
                decision_engine=self.decision_engine,
                planner=self.planner,
                execution_engine=self.execution_engine,
                verifier=self.verifier,
                composer=self.composer,
                memory_service=self.memory_service,
                web_provider=self.web_provider,
                recovery_engine=self.recovery_engine,
            )

    def _sync_runtime(self) -> None:
        """Synchronize runtime component bindings with current pipeline attributes."""
        self.runtime.understanding = self.understanding
        self.runtime.decision_engine = self.decision_engine
        self.runtime.planner = self.planner
        self.runtime.execution_engine = self.execution_engine
        self.runtime.verifier = self.verifier
        self.runtime.composer = self.composer
        self.runtime.memory_service = self.memory_service
        self.runtime.web_provider = self.web_provider
        self.runtime.recovery_engine = self.recovery_engine

    def _inject_context(
        self,
        plan: Plan,
        request: Request,
        history_entries: List[Any],
        is_empty: bool,
    ) -> None:
        """Inject contextual dependencies (memory, history, web) into plan steps without mutating Request."""
        if not plan.steps or is_empty:
            return

        formatted_history: List[Dict[str, str]] = [
            {"role": msg.role.value, "content": msg.content}
            for msg in history_entries
        ]
        for step in plan.steps:
            step_type = getattr(step, "type", None)
            step_tool = getattr(step, "tool", None)
            if step_type == "chat" or step_tool == "chat":
                params = dict(step.parameters) if step.parameters else {}
                if "query" not in params:
                    params["query"] = request.original_text
                if "history" not in params:
                    params["history"] = formatted_history
                step.parameters = params
            elif step_type == "memory" or step_tool == "memory":
                params = dict(step.parameters) if step.parameters else {}
                if "memory_service" not in params and self.memory_service is not None:
                    params["memory_service"] = self.memory_service
                if "user_id" not in params:
                    params["user_id"] = "default_user"
                step.parameters = params
            elif step_type == "web" or step_tool == "web":
                params = dict(step.parameters) if step.parameters else {}
                if "web_provider" not in params and self.web_provider is not None:
                    params["web_provider"] = self.web_provider
                step.parameters = params

    def _execute_plan(
        self,
        plan: Plan,
        request: Request,
        history_entries: List[Any],
        is_empty: bool,
    ) -> List[Result]:
        """Inject contextual dependencies and execute plan via execution_engine."""
        self._inject_context(plan, request, history_entries, is_empty)
        return self.execution_engine.execute(plan)

    def process(self, input_data: Any) -> PipelineResult:
        """
        Execute the full cognitive turn lifecycle backed by CognitiveRuntime
        and return structured PipelineResult for backward compatibility.
        """
        self._sync_runtime()
        turn_result = self.runtime.execute_turn(input_data)
        return turn_result.to_pipeline_result()


    def run(self, input_data: Any) -> str:
        """
        Execute the full Phase 2 lifecycle and return only the final response string.
        Delegates directly to process().
        """
        return self.process(input_data).response
