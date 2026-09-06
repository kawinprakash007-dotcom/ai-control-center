from typing import Optional, Dict, Set, Any, List, Tuple
import time

from core.models.reasoning import (
    ActionProposal,
    ProposalValidationResult,
)
from core.models.tool_call import ToolCall
from core.models.perception import VisualScene
from core.interfaces.reasoning_interface import ActionProposalValidatorInterface
from tools.tool_orchestrator import DEFAULT_ALLOWED_CAPABILITIES, PROHIBITED_STRINGS, _is_prohibited


class ActionProposalValidator(ActionProposalValidatorInterface):
    """
    Deterministic validation boundary ensuring that ActionProposals from reasoning
    providers are structurally valid, non-prohibited, schema-compliant, and fresh
    before converting into executable ToolCalls.

    Safety Principles:
    1. A model proposes intent; ATLAS decides if it is valid and safe.
    2. Invalid proposals fail closed and return structured validation errors.
    3. Stale visual targets are rejected.
    4. Prohibited strings (shell, exec, credential theft, etc.) are deterministically blocked.
    """

    def __init__(
        self,
        allowed_capabilities: Optional[Dict[str, Set[str]]] = None,
        max_target_age_seconds: float = 30.0,
    ):
        self.allowed_capabilities = (
            {k.lower(): {a.lower() for a in v} for k, v in allowed_capabilities.items()}
            if allowed_capabilities is not None
            else {k: set(v) for k, v in DEFAULT_ALLOWED_CAPABILITIES.items()}
        )
        self.max_target_age_seconds = max_target_age_seconds

    def validate(
        self,
        proposal: ActionProposal,
        current_scene: Optional[VisualScene] = None,
        allowed_capabilities: Optional[Dict[str, Set[str]]] = None,
    ) -> ProposalValidationResult:
        errors: List[str] = []
        is_stale_target = False
        is_unknown_cap = False
        is_unknown_act = False
        is_prohibited_act = False

        active_allowed = (
            {k.lower(): {a.lower() for a in v} for k, v in allowed_capabilities.items()}
            if allowed_capabilities is not None
            else self.allowed_capabilities
        )

        # 1. Structural requirements
        if not proposal.proposal_id:
            errors.append("Missing proposal_id")
        if not proposal.goal_reference:
            errors.append("Missing goal_reference")
        if not proposal.capability:
            errors.append("Missing capability")
            is_unknown_cap = True
        if not proposal.action:
            errors.append("Missing action")
            is_unknown_act = True

        if errors:
            return ProposalValidationResult(
                is_valid=False,
                errors=tuple(errors),
                unknown_capability=is_unknown_cap,
                unknown_action=is_unknown_act,
            )

        cap = proposal.capability.lower()
        act = proposal.action.lower()

        # 2. Check for prohibited security terms (shell, powershell, exec, cmd, credential theft)
        if _is_prohibited(cap, PROHIBITED_STRINGS) or _is_prohibited(act, PROHIBITED_STRINGS):
            errors.append(f"Prohibited security term detected in capability '{cap}' or action '{act}'")
            is_prohibited_act = True
            return ProposalValidationResult(
                is_valid=False,
                errors=tuple(errors),
                prohibited_action=True,
            )

        # Check parameter values for prohibited injection strings
        for param_k, param_v in proposal.parameters.items():
            if _is_prohibited(str(param_k), PROHIBITED_STRINGS):
                errors.append(f"Prohibited key '{param_k}' in proposal parameters")
                is_prohibited_act = True
            if isinstance(param_v, str) and _is_prohibited(param_v, PROHIBITED_STRINGS):
                errors.append(f"Prohibited string '{param_v}' in proposal parameter value")
                is_prohibited_act = True

        if is_prohibited_act:
            return ProposalValidationResult(
                is_valid=False,
                errors=tuple(errors),
                prohibited_action=True,
            )

        # 3. Whitelist capability check
        if cap not in active_allowed:
            errors.append(f"Unknown or unauthorized capability: '{cap}'. Allowed: {sorted(active_allowed.keys())}")
            is_unknown_cap = True

        # 4. Whitelist action check
        elif act not in active_allowed[cap]:
            errors.append(f"Unknown or unauthorized action '{act}' for capability '{cap}'. Allowed: {sorted(active_allowed[cap])}")
            is_unknown_act = True

        # 5. Parameter schema validation per capability
        if not errors:
            param_errors = self._validate_action_params(cap, act, proposal.parameters)
            errors.extend(param_errors)

        # 6. Visual target and provenance freshness check
        if not errors and cap == "computer":
            scene_errors, stale = self._validate_visual_provenance(proposal, current_scene)
            if stale:
                is_stale_target = True
            errors.extend(scene_errors)

        if errors:
            return ProposalValidationResult(
                is_valid=False,
                errors=tuple(errors),
                stale_target=is_stale_target,
                unknown_capability=is_unknown_cap,
                unknown_action=is_unknown_act,
                prohibited_action=is_prohibited_act,
            )

        # 7. Valid -> Construct governed ToolCall
        tool_call = ToolCall(
            capability=cap,
            action=act,
            parameters=dict(proposal.parameters),
            reason=proposal.rationale,
            call_id=proposal.proposal_id,
        )

        return ProposalValidationResult(
            is_valid=True,
            tool_call=tool_call,
            errors=(),
            stale_target=False,
            unknown_capability=False,
            unknown_action=False,
            prohibited_action=False,
            metadata={"validated_at": time.time(), "goal_reference": proposal.goal_reference},
        )

    def _validate_action_params(self, capability: str, action: str, parameters: Dict[str, Any]) -> List[str]:
        errs: List[str] = []
        if capability == "computer":
            if action in ("click", "double_click", "move"):
                x = parameters.get("x")
                y = parameters.get("y")
                if x is None or y is None:
                    errs.append(f"Action '{action}' requires non-null 'x' and 'y' integer coordinates")
                else:
                    if not isinstance(x, int) or not isinstance(y, int) or x < 0 or y < 0:
                        errs.append(f"Action '{action}' coordinates must be non-negative integers: got ({x}, {y})")
            elif action == "type":
                text = parameters.get("text")
                if text is None or not isinstance(text, str) or not text:
                    errs.append("Action 'type' requires non-empty string parameter 'text'")
            elif action == "press_key":
                key = parameters.get("key")
                if not key or not isinstance(key, str):
                    errs.append("Action 'press_key' requires non-empty string parameter 'key'")
            elif action == "scroll":
                amount = parameters.get("amount")
                if amount is None or not isinstance(amount, int):
                    errs.append("Action 'scroll' requires integer parameter 'amount'")
            elif action == "wait":
                seconds = parameters.get("seconds", parameters.get("duration"))
                if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
                    errs.append("Action 'wait' requires non-negative numeric 'seconds'")
        elif capability == "web":
            if action in ("search", "research"):
                query = parameters.get("query")
                if not query or not isinstance(query, str) or not query.strip():
                    errs.append(f"Web '{action}' requires non-empty string parameter 'query'")
            elif action == "fetch":
                url = parameters.get("url")
                if not url or not isinstance(url, str) or not url.strip():
                    errs.append("Web 'fetch' requires non-empty string parameter 'url'")
        elif capability == "memory":
            if action in ("save", "remember"):
                if "key" not in parameters or "value" not in parameters:
                    errs.append(f"Memory '{action}' requires 'key' and 'value' parameters")
            elif action in ("read", "recall", "forget"):
                if "key" not in parameters:
                    errs.append(f"Memory '{action}' requires 'key' parameter")
        elif capability == "knowledge":
            if action in ("query", "retrieve", "search"):
                query = parameters.get("query")
                if not query or not isinstance(query, str) or not query.strip():
                    errs.append(f"Knowledge '{action}' requires non-empty string parameter 'query'")

        return errs

    def _validate_visual_provenance(
        self,
        proposal: ActionProposal,
        current_scene: Optional[VisualScene],
    ) -> Tuple[List[str], bool]:
        errs: List[str] = []
        is_stale = False

        # If action requires visual interaction coordinates on desktop
        if proposal.action in ("click", "double_click", "move"):
            if proposal.observation_reference:
                if current_scene is None:
                    errs.append(
                        f"Proposal references observation '{proposal.observation_reference}', but no current VisualScene is available"
                    )
                    is_stale = True
                elif current_scene.source_observation_id != proposal.observation_reference:
                    errs.append(
                        f"Observation mismatch: proposal references observation '{proposal.observation_reference}', but current scene is '{current_scene.source_observation_id}'"
                    )
                    is_stale = True

            # If a specific target reference was specified, verify it exists in current scene
            if proposal.target_reference and current_scene is not None:
                el = current_scene.get_element_by_id(proposal.target_reference)
                if el is None:
                    errs.append(f"Target element '{proposal.target_reference}' does not exist in current VisualScene")
                    is_stale = True

        return errs, is_stale
