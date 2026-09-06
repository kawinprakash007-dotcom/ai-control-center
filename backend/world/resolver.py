import time
import uuid
from typing import Optional

from core.interfaces.world_interface import ConflictResolverInterface
from core.models.world_state import (
    ConflictPolicy,
    ConflictResolution,
    ResolutionStrategy,
    StateConflict,
)


class DeterministicConflictResolver(ConflictResolverInterface):
    """
    Deterministic, model-neutral conflict resolver for world state.
    Applies explicit authority hierarchies, confidence thresholds, and recency rules.
    Never executes models, prompts, or tools.
    """

    def resolve(
        self,
        conflict: StateConflict,
        policy: Optional[ConflictPolicy] = None,
    ) -> Optional[ConflictResolution]:
        effective_policy = policy or ConflictPolicy()
        existing = conflict.existing_condition
        competing = conflict.competing_observation

        # 1. Source Authority Evaluation
        auth_existing = effective_policy.get_authority(existing.provenance.source_type)
        auth_competing = effective_policy.get_authority(competing.source_type)

        if auth_competing > auth_existing:
            return ConflictResolution(
                resolution_id=f"res_{uuid.uuid4().hex[:12]}",
                conflict_id=conflict.conflict_id,
                resolved_value=competing.value,
                winning_source=competing.source_id,
                strategy=ResolutionStrategy.SOURCE_AUTHORITY,
                rationale=(
                    f"Competing source '{competing.source_type}' (auth={auth_competing}) "
                    f"supersedes existing source '{existing.provenance.source_type}' (auth={auth_existing})."
                ),
            )
        elif auth_existing > auth_competing:
            return ConflictResolution(
                resolution_id=f"res_{uuid.uuid4().hex[:12]}",
                conflict_id=conflict.conflict_id,
                resolved_value=existing.value,
                winning_source=existing.provenance.source_id,
                strategy=ResolutionStrategy.SOURCE_AUTHORITY,
                rationale=(
                    f"Existing source '{existing.provenance.source_type}' (auth={auth_existing}) "
                    f"retains precedence over competing source '{competing.source_type}' (auth={auth_competing})."
                ),
            )

        # 2. Confidence Delta Evaluation (Equal Authority)
        if effective_policy.confidence_tie_breaker:
            conf_delta = competing.confidence - existing.confidence
            if conf_delta > effective_policy.confidence_delta_threshold:
                return ConflictResolution(
                    resolution_id=f"res_{uuid.uuid4().hex[:12]}",
                    conflict_id=conflict.conflict_id,
                    resolved_value=competing.value,
                    winning_source=competing.source_id,
                    strategy=ResolutionStrategy.HIGHER_CONFIDENCE,
                    rationale=(
                        f"Competing observation confidence ({competing.confidence:.2f}) exceeds "
                        f"existing confidence ({existing.confidence:.2f}) by threshold ({conf_delta:.2f} > {effective_policy.confidence_delta_threshold:.2f})."
                    ),
                )
            elif -conf_delta > effective_policy.confidence_delta_threshold:
                return ConflictResolution(
                    resolution_id=f"res_{uuid.uuid4().hex[:12]}",
                    conflict_id=conflict.conflict_id,
                    resolved_value=existing.value,
                    winning_source=existing.provenance.source_id,
                    strategy=ResolutionStrategy.HIGHER_CONFIDENCE,
                    rationale=(
                        f"Existing condition confidence ({existing.confidence:.2f}) exceeds "
                        f"competing confidence ({competing.confidence:.2f}) by threshold ({-conf_delta:.2f} > {effective_policy.confidence_delta_threshold:.2f})."
                    ),
                )

        # 3. Recency Evaluation (Equal Authority and Equivalent Confidence)
        if effective_policy.recency_tie_breaker:
            if competing.timestamp > existing.observed_at:
                return ConflictResolution(
                    resolution_id=f"res_{uuid.uuid4().hex[:12]}",
                    conflict_id=conflict.conflict_id,
                    resolved_value=competing.value,
                    winning_source=competing.source_id,
                    strategy=ResolutionStrategy.RECENCY,
                    rationale=(
                        f"Competing observation is more recent ({competing.timestamp:.2f} > {existing.observed_at:.2f})."
                    ),
                )
            elif existing.observed_at > competing.timestamp:
                return ConflictResolution(
                    resolution_id=f"res_{uuid.uuid4().hex[:12]}",
                    conflict_id=conflict.conflict_id,
                    resolved_value=existing.value,
                    winning_source=existing.provenance.source_id,
                    strategy=ResolutionStrategy.RECENCY,
                    rationale=(
                        f"Existing condition timestamp is newer ({existing.observed_at:.2f} > {competing.timestamp:.2f})."
                    ),
                )

        # 4. Tie / Ambiguous Conflict
        return None
