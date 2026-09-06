import re
from typing import List, Optional, Set, Tuple

from core.interfaces.goal_interface import GoalDecomposerInterface
from core.models.goal import Goal, Objective, ObjectiveStatus


class DecomposerError(ValueError):
    """Raised when goal decomposition fails due to malformed or invalid goal data."""
    pass


class DeterministicGoalDecomposer(GoalDecomposerInterface):
    """
    Model-neutral, deterministic decomposition engine.
    Splits multi-part goals into bounded, ordered Objective instances with explicit dependencies.
    Operates without invoking external LLMs or autonomous agent loops.
    """

    DEFAULT_MAX_OBJECTIVES = 20

    def __init__(self, max_objectives: int = DEFAULT_MAX_OBJECTIVES):
        self.max_objectives = max_objectives

    def decompose(self, goal: Goal) -> Tuple[Objective, ...]:
        if not isinstance(goal, Goal):
            raise DecomposerError(f"Expected Goal instance, got {type(goal).__name__}")

        raw_text = (goal.original_goal or "").strip()
        if not raw_text:
            raise DecomposerError("Cannot decompose empty goal.")

        # If goal already has pre-configured objectives, validate and return them
        if goal.objectives:
            self._validate_objectives(goal.objectives)
            return goal.objectives

        # Parse steps from text
        steps = self._extract_steps(raw_text)

        if not steps:
            steps = [raw_text]

        if len(steps) > self.max_objectives:
            raise DecomposerError(
                f"Decomposition produced {len(steps)} objectives, exceeding limit of {self.max_objectives}."
            )

        # Detect duplicates
        seen_signatures: Set[str] = set()
        objectives: List[Objective] = []

        for idx, step_desc in enumerate(steps, 1):
            clean_desc = step_desc.strip()
            if not clean_desc:
                continue

            sig = clean_desc.lower()
            if sig in seen_signatures:
                raise DecomposerError(f"Duplicate objective detected in goal decomposition: '{clean_desc}'")
            seen_signatures.add(sig)

            obj_id = f"{goal.goal_id}_obj_{idx}"
            # Linear sequential dependency: step N depends on step N-1
            deps = (f"{goal.goal_id}_obj_{idx - 1}",) if idx > 1 else ()
            initial_status = ObjectiveStatus.READY if idx == 1 else ObjectiveStatus.PENDING

            obj = Objective(
                objective_id=obj_id,
                description=clean_desc,
                order=idx,
                dependencies=deps,
                status=initial_status,
                max_attempts=goal.constraints.max_turns_per_objective if goal.constraints else 3,
            )
            objectives.append(obj)

        if not objectives:
            raise DecomposerError("Decomposition resulted in zero valid objectives.")

        self._validate_objectives(tuple(objectives))
        return tuple(objectives)

    def _extract_steps(self, text: str) -> List[str]:
        """Extract steps using numbered lists, bullet points, or sequential connectives."""
        # 1. Check for numbered lines: e.g. "1. Do X\n2. Do Y"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        numbered_pattern = re.compile(r"^(?:(?:step\s+)?\d+[\.\)]|\-|\*)\s*(.+)$", re.IGNORECASE)

        extracted_lines = []
        for line in lines:
            match = numbered_pattern.match(line)
            if match:
                extracted_lines.append(match.group(1).strip())

        if len(extracted_lines) >= 2:
            return extracted_lines

        # 2. Check for semicolon separation: e.g. "First do X; then do Y; finally do Z"
        if ";" in text:
            parts = [p.strip() for p in text.split(";") if p.strip()]
            if len(parts) >= 2:
                return parts

        # 3. Check for explicit sequential keywords: "then", "after that", "afterwards"
        keyword_pattern = re.compile(r"\b(?:then|after that|afterward|afterwards)\b", re.IGNORECASE)
        parts = keyword_pattern.split(text)
        if len(parts) >= 2:
            cleaned = [p.strip().rstrip(",.") for p in parts if p.strip().rstrip(",.")]
            if len(cleaned) >= 2:
                return cleaned

        return [text]

    def _validate_objectives(self, objectives: Tuple[Objective, ...]) -> None:
        """Validate dependency graph for circular references and invalid links."""
        obj_ids = {obj.objective_id for obj in objectives}
        for obj in objectives:
            for dep in obj.dependencies:
                if dep not in obj_ids:
                    raise DecomposerError(
                        f"Objective '{obj.objective_id}' references non-existent dependency '{dep}'."
                    )
                if dep == obj.objective_id:
                    raise DecomposerError(
                        f"Objective '{obj.objective_id}' cannot depend on itself."
                    )
