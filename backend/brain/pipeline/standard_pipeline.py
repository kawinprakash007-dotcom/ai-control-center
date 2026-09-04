from typing import Any, Optional

from core.interfaces.pipeline_interface import PipelineInterface
from core.interfaces.request_understanding_interface import RequestUnderstandingInterface
from core.interfaces.decision_engine_interface import DecisionEngineInterface
from core.interfaces.decision_planner_interface import DecisionPlannerInterface
from core.interfaces.execution_engine_interface import ExecutionEngineInterface
from core.interfaces.verification_interface import VerificationInterface
from core.interfaces.response_composer_interface import ResponseComposerInterface
from core.models.pipeline import PipelineResult

from brain.request_understanding import StandardRequestUnderstanding
from brain.decision_engine import StandardDecisionEngine
from brain.planning import StandardPlanner
from brain.execution import StandardExecutionEngine
from brain.verification import StandardVerifier
from brain.response import StandardResponseComposer


class StandardPipeline(PipelineInterface):
    """
    Deterministic implementation of PipelineInterface.
    Orchestrates the 6-stage Phase 2 request lifecycle:
    Understanding -> Decision -> Planning -> Execution -> Verification -> Response.
    """

    def __init__(
        self,
        understanding: Optional[RequestUnderstandingInterface] = None,
        decision_engine: Optional[DecisionEngineInterface] = None,
        planner: Optional[DecisionPlannerInterface] = None,
        execution_engine: Optional[ExecutionEngineInterface] = None,
        verifier: Optional[VerificationInterface] = None,
        composer: Optional[ResponseComposerInterface] = None,
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

    def process(self, input_data: Any) -> PipelineResult:
        """
        Execute the full Phase 2 lifecycle and return structured PipelineResult.
        """
        request = self.understanding.understand(input_data)
        decision = self.decision_engine.decide(request)
        plan = self.planner.plan(decision)
        results = self.execution_engine.execute(plan)
        verification = self.verifier.verify(plan, results)
        response = self.composer.compose(
            request,
            decision,
            plan,
            results,
            verification,
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
