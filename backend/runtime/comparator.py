from typing import Any, Dict, List, Optional, Tuple

from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    CognitiveTrace,
    TurnStatus,
)
from core.models.replay import (
    ComparisonResult,
    DivergenceSeverity,
    ReplayDivergence,
    ReplayDivergenceType,
)


class TraceComparator:
    """
    Deterministic comparator evaluating behavioral and semantic divergence
    between an original CognitiveTrace and a replayed CognitiveTrace.
    Ignores non-semantic variance (exact timestamps, wall-clock durations, ephemeral run IDs).
    """

    @classmethod
    def compare(
        cls,
        original_trace: CognitiveTrace,
        replay_trace: CognitiveTrace,
        max_divergences: int = 50,
    ) -> ComparisonResult:
        """
        Perform a sequential semantic comparison between original and replay traces.
        """
        orig_events = original_trace.events
        repl_events = replay_trace.events
        divergences: List[ReplayDivergence] = []
        matched_count = 0

        max_len = max(len(orig_events), len(repl_events))

        for idx in range(max_len):
            if len(divergences) >= max_divergences:
                break

            # Check if original event is missing in replay
            if idx >= len(repl_events):
                orig_ev = orig_events[idx]
                divergences.append(
                    ReplayDivergence(
                        event_index=idx,
                        stage=orig_ev.stage,
                        divergence_type=ReplayDivergenceType.MISSING_EVENT,
                        expected=f"{orig_ev.stage.value}:{orig_ev.event_type.value}",
                        actual=None,
                        severity=DivergenceSeverity.HIGH,
                        reason=f"Replay trace ended prematurely; missing original event {orig_ev.event_type.value}",
                    )
                )
                continue

            # Check if replay produced extra unexpected events
            if idx >= len(orig_events):
                repl_ev = repl_events[idx]
                divergences.append(
                    ReplayDivergence(
                        event_index=idx,
                        stage=repl_ev.stage,
                        divergence_type=ReplayDivergenceType.EXTRA_EVENT,
                        expected=None,
                        actual=f"{repl_ev.stage.value}:{repl_ev.event_type.value}",
                        severity=DivergenceSeverity.HIGH,
                        reason=f"Replay produced extra unexpected event {repl_ev.event_type.value}",
                    )
                )
                continue

            orig_ev = orig_events[idx]
            repl_ev = repl_events[idx]

            # 1. Compare stage
            if orig_ev.stage != repl_ev.stage:
                divergences.append(
                    ReplayDivergence(
                        event_index=idx,
                        stage=repl_ev.stage,
                        divergence_type=ReplayDivergenceType.STAGE_MISMATCH,
                        expected=orig_ev.stage.value,
                        actual=repl_ev.stage.value,
                        severity=DivergenceSeverity.HIGH,
                        reason=f"Stage mismatch: expected {orig_ev.stage.value}, got {repl_ev.stage.value}",
                    )
                )
                continue

            # 2. Compare event type
            if orig_ev.event_type != repl_ev.event_type:
                divergences.append(
                    ReplayDivergence(
                        event_index=idx,
                        stage=repl_ev.stage,
                        divergence_type=ReplayDivergenceType.STAGE_MISMATCH,
                        expected=orig_ev.event_type.value,
                        actual=repl_ev.event_type.value,
                        severity=DivergenceSeverity.HIGH,
                        reason=f"Event type mismatch: expected {orig_ev.event_type.value}, got {repl_ev.event_type.value}",
                    )
                )
                continue

            # 3. Compare status (e.g. OK vs FAILED)
            if orig_ev.status != repl_ev.status:
                divergences.append(
                    ReplayDivergence(
                        event_index=idx,
                        stage=repl_ev.stage,
                        divergence_type=ReplayDivergenceType.RESULT_MISMATCH,
                        expected=orig_ev.status,
                        actual=repl_ev.status,
                        severity=DivergenceSeverity.HIGH,
                        reason=f"Event status mismatch: expected {orig_ev.status}, got {repl_ev.status}",
                    )
                )
                continue

            # 4. Stage-specific semantic checks
            meta_orig = orig_ev.metadata or {}
            meta_repl = repl_ev.metadata or {}

            # Policy decision comparison
            if orig_ev.event_type == CognitiveEventType.POLICY_DECIDED:
                p_orig = meta_orig.get("decision")
                p_repl = meta_repl.get("decision")
                if p_orig and p_repl and p_orig != p_repl:
                    divergences.append(
                        ReplayDivergence(
                            event_index=idx,
                            stage=CognitiveStage.POLICY,
                            divergence_type=ReplayDivergenceType.POLICY_MISMATCH,
                            expected=p_orig,
                            actual=p_repl,
                            severity=DivergenceSeverity.CRITICAL,
                            reason=f"Policy decision mismatch: expected {p_orig}, got {p_repl}",
                        )
                    )
                    continue

            # Model provider routing comparison
            if orig_ev.event_type == CognitiveEventType.MODEL_ROUTED:
                prov_orig = meta_orig.get("provider_id")
                prov_repl = meta_repl.get("provider_id")
                if prov_orig and prov_repl and prov_orig != prov_repl:
                    divergences.append(
                        ReplayDivergence(
                            event_index=idx,
                            stage=CognitiveStage.ROUTING,
                            divergence_type=ReplayDivergenceType.PROVIDER_MISMATCH,
                            expected=prov_orig,
                            actual=prov_repl,
                            severity=DivergenceSeverity.MEDIUM,
                            reason=f"Model provider mismatch: expected {prov_orig}, got {prov_repl}",
                        )
                    )
                    continue

            # Tool execution success comparison
            if orig_ev.event_type == CognitiveEventType.TOOL_EXECUTED:
                succ_orig = meta_orig.get("success")
                succ_repl = meta_repl.get("success")
                if succ_orig is not None and succ_repl is not None and succ_orig != succ_repl:
                    divergences.append(
                        ReplayDivergence(
                            event_index=idx,
                            stage=CognitiveStage.EXECUTION,
                            divergence_type=ReplayDivergenceType.RESULT_MISMATCH,
                            expected=succ_orig,
                            actual=succ_repl,
                            severity=DivergenceSeverity.HIGH,
                            reason=f"Tool execution success mismatch: expected {succ_orig}, got {succ_repl}",
                        )
                    )
                    continue

            matched_count += 1

        # Check final turn status equivalence
        if original_trace.final_status != replay_trace.final_status:
            divergences.append(
                ReplayDivergence(
                    event_index=max_len,
                    stage=CognitiveStage.COMPLETED,
                    divergence_type=ReplayDivergenceType.STATE_MISMATCH,
                    expected=original_trace.final_status.value,
                    actual=replay_trace.final_status.value,
                    severity=DivergenceSeverity.HIGH,
                    reason=f"Final turn status mismatch: expected {original_trace.final_status.value}, got {replay_trace.final_status.value}",
                )
            )

        # Determine overall comparison status
        if not divergences:
            status = "IDENTICAL"
            summary = f"Replay perfectly matched original trace ({matched_count} events)."
        else:
            # Check if only LOW severity divergences exist
            critical_or_high = any(
                d.severity in (DivergenceSeverity.HIGH, DivergenceSeverity.CRITICAL)
                for d in divergences
            )
            if not critical_or_high:
                status = "SEMANTICALLY_EQUIVALENT"
                summary = f"Replay semantically equivalent with {len(divergences)} non-critical divergence(s)."
            else:
                status = "DIVERGENT"
                summary = f"Replay diverged with {len(divergences)} semantic divergence(s)."

        return ComparisonResult(
            status=status,
            divergences=tuple(divergences),
            matched_events_count=matched_count,
            total_original_events=len(orig_events),
            total_replay_events=len(repl_events),
            summary=summary,
        )


def format_debug_summary(comparison: ComparisonResult, source_turn_id: str) -> str:
    """
    Format a clean, human-readable diagnostic report from a comparison result.
    Does not print raw secrets, tokens, or unbounded image payloads.
    """
    lines = [
        "==================================================",
        "ATLAS REPLAY DIAGNOSTIC SUMMARY",
        "==================================================",
        f"SOURCE TURN ID: {source_turn_id}",
        f"REPLAY STATUS:  {comparison.status}",
        f"MATCHED EVENTS: {comparison.matched_events_count}/{comparison.total_original_events} (original) | {comparison.total_replay_events} (replay)",
        f"SUMMARY:        {comparison.summary}",
    ]

    if comparison.divergences:
        lines.append("--------------------------------------------------")
        lines.append(f"DIVERGENCES ({len(comparison.divergences)}):")
        for i, div in enumerate(comparison.divergences, 1):
            lines.append(f"  [{i}] Index {div.event_index} | Stage: {div.stage.value} | Type: {div.divergence_type.value} | Severity: {div.severity.value}")
            lines.append(f"      Expected: {div.expected}")
            lines.append(f"      Actual:   {div.actual}")
            if div.reason:
                lines.append(f"      Reason:   {div.reason}")
    lines.append("==================================================")
    return "\n".join(lines)
