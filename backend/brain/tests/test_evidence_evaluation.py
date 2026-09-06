import pytest
from datetime import datetime, timezone, timedelta

from core.models.web import EvidenceItem, EvidenceSet
from core.models.research import EvidenceAssessment, ResearchGap, Contradiction
from web.evidence_evaluator import EvidenceEvaluator


# ============================================================================
# EVIDENCE ASSESSMENT & HEURISTIC SIGNAL TESTS
# ============================================================================

def test_evidence_assessment_creation_and_serialization():
    """1. EvidenceAssessment instantiates and serializes cleanly."""
    assessment = EvidenceAssessment(
        evidence_id="ev-123",
        relevance_score=0.85,
        freshness_score=0.90,
        coverage_score=0.75,
        quality_score=0.80,
        has_contradiction=False,
        source_signal="documentation",
        notes="High quality official docs",
    )
    d = assessment.to_dict()
    assert d["evidence_id"] == "ev-123"
    assert d["relevance_score"] == 0.85
    assert d["source_signal"] == "documentation"

    rebuilt = EvidenceAssessment.from_dict(d)
    assert rebuilt.evidence_id == "ev-123"
    assert rebuilt.relevance_score == 0.85
    assert rebuilt.has_contradiction is False


def test_relevance_scoring():
    """2. Relevance scoring reflects keyword and title overlap with objective."""
    evaluator = EvidenceEvaluator()
    objective = "Raspberry Pi 5 AI performance benchmark"

    # Highly relevant item
    high_item = EvidenceItem(
        id="ev-high",
        title="Raspberry Pi 5 AI Performance Benchmark Results",
        url="https://example.com/rpi5-benchmarks",
        domain="example.com",
        content="Detailed benchmark numbers for Hailo AI accelerator on Raspberry Pi 5.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    score_high = evaluator.calculate_relevance(high_item, objective)
    assert score_high >= 0.7

    # Irrelevant item
    low_item = EvidenceItem(
        id="ev-low",
        title="Gardening Tips for Spring Tomatoes",
        url="https://garden.org/tomatoes",
        domain="garden.org",
        content="How to fertilize soil and prune young tomato plants.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    score_low = evaluator.calculate_relevance(low_item, objective)
    assert score_low <= 0.2
    assert score_high > score_low


def test_freshness_signal():
    """3. Freshness scoring scores recent timestamps and modern year references higher."""
    evaluator = EvidenceEvaluator()

    # Recent item
    recent_item = EvidenceItem(
        id="ev-recent",
        title="Latest AI Hardware 2026",
        url="https://news.com/2026",
        domain="news.com",
        content="Current benchmarks from 2026 hardware releases.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    freshness_recent = evaluator.calculate_freshness(recent_item)
    assert freshness_recent >= 0.8

    # Old item from 5 years ago
    old_time = (datetime.now(timezone.utc) - timedelta(days=1800)).isoformat()
    old_item = EvidenceItem(
        id="ev-old",
        title="Old Hardware Analysis",
        url="https://archive.com/2019",
        domain="archive.com",
        content="Legacy metrics from 2019.",
        retrieved_at=old_time,
    )
    freshness_old = evaluator.calculate_freshness(old_item)
    assert freshness_recent > freshness_old


def test_source_quality_signals():
    """4. Source quality evaluates HTTPS, length, and authoritative documentation signals."""
    evaluator = EvidenceEvaluator()

    # Authoritative doc source over HTTPS
    doc_item = EvidenceItem(
        id="ev-doc",
        title="Official Hardware Specifications",
        url="https://datasheets.raspberrypi.com/rpi5/spec.pdf",
        domain="datasheets.raspberrypi.com",
        content="Detailed technical specifications of the BCM2712 application processor and PCIe interface.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    doc_signal = evaluator.detect_source_signal(doc_item)
    assert doc_signal in ("documentation", "official")
    quality_doc = evaluator.calculate_quality(doc_item)
    assert quality_doc >= 0.7

    # Insecure, stub content
    stub_item = EvidenceItem(
        id="ev-stub",
        title="Hi",
        url="http://unknown.org/stub",
        domain="unknown.org",
        content="short",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    quality_stub = evaluator.calculate_quality(stub_item)
    assert quality_stub < quality_doc


def test_duplicate_evidence_handling():
    """5. EvidenceEvaluator processes evidence without crashing on duplicate or overlapping items."""
    evaluator = EvidenceEvaluator()
    items = [
        EvidenceItem(
            id="ev-1",
            title="Title A",
            url="https://example.com/a",
            domain="example.com",
            content="Content A",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        ),
        EvidenceItem(
            id="ev-2",
            title="Title A",
            url="https://example.com/a",
            domain="example.com",
            content="Content A",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        ),
    ]
    assessments, gaps, contradictions = evaluator.evaluate(items, "Topic A")
    assert len(assessments) == 2
    assert assessments[0].evidence_id == "ev-1"
    assert assessments[1].evidence_id == "ev-2"


# ============================================================================
# COVERAGE & RESEARCH GAP TESTS
# ============================================================================

def test_coverage_calculation_and_research_gaps():
    """6 & 7. Coverage evaluation detects missing objective aspects and creates ResearchGaps."""
    evaluator = EvidenceEvaluator()
    objective = "Compare Raspberry Pi 5 and Jetson Orin Nano for AI performance"

    # Only Raspberry Pi evidence present
    partial_evidence = [
        EvidenceItem(
            id="ev-rpi",
            title="Raspberry Pi 5 AI Performance",
            url="https://example.com/rpi5",
            domain="example.com",
            content="Raspberry Pi 5 benchmark numbers with Hailo-8L accelerator.",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
    ]
    coverage, gaps = evaluator.calculate_coverage(partial_evidence, objective)

    assert coverage < 1.0
    assert len(gaps) >= 1
    # Check that Jetson Orin Nano is flagged as a missing gap
    gap_topics = [g.topic.lower() for g in gaps]
    assert any("jetson" in t for t in gap_topics)
    assert gaps[0].priority in (1, 2)


def test_complete_coverage():
    """Coverage reaches 1.0 when all aspects are addressed in evidence."""
    evaluator = EvidenceEvaluator()
    objective = "Compare Raspberry Pi 5 and Jetson Orin Nano"

    full_evidence = [
        EvidenceItem(
            id="ev-rpi",
            title="Raspberry Pi 5 AI",
            url="https://example.com/rpi5",
            domain="example.com",
            content="Raspberry Pi 5 benchmark analysis.",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        ),
        EvidenceItem(
            id="ev-jetson",
            title="Jetson Orin Nano AI",
            url="https://example.com/jetson",
            domain="example.com",
            content="Jetson Orin Nano benchmarks and specifications.",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        ),
    ]
    coverage, gaps = evaluator.calculate_coverage(full_evidence, objective)
    assert coverage >= 0.9
    assert len(gaps) == 0


# ============================================================================
# CONTRADICTION DETECTION TESTS
# ============================================================================

def test_contradiction_detection_polarity():
    """8. Detects factual contradiction with opposing polarity (supports vs does not support)."""
    evaluator = EvidenceEvaluator()
    item_a = EvidenceItem(
        id="ev-pro",
        title="Source A: PCIe 3.0 Support on Pi 5",
        url="https://source-a.org/rpi5-pcie",
        domain="source-a.org",
        content="The board supports PCIe 3.0 speeds reliably with proper configuration.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    item_b = EvidenceItem(
        id="ev-con",
        title="Source B: PCIe 3.0 Compatibility",
        url="https://source-b.org/rpi5-pcie",
        domain="source-b.org",
        content="The platform does not support PCIe 3.0 and is incompatible with Gen3 certified cables.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )

    contradictions = evaluator.detect_contradictions([item_a, item_b], "PCIe 3.0 support on Raspberry Pi 5")
    assert len(contradictions) >= 1
    contra = contradictions[0]
    assert "ev-pro" in contra.evidence_ids
    assert "ev-con" in contra.evidence_ids
    assert len(contra.conflicting_claims) == 2


def test_contradiction_detection_metric_divergence():
    """Detects numeric divergence on matching measurement units."""
    evaluator = EvidenceEvaluator()
    item_a = EvidenceItem(
        id="ev-tops-13",
        title="Hailo Module Performance",
        url="https://source-a.org/tops",
        domain="source-a.org",
        content="The Hailo-8L accelerator reaches 13 TOPS computing throughput.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    item_b = EvidenceItem(
        id="ev-tops-26",
        title="Hailo Module Hardware Specs",
        url="https://source-b.org/tops",
        domain="source-b.org",
        content="The module delivers up to 26 TOPS maximum computing throughput.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )

    contradictions = evaluator.detect_contradictions([item_a, item_b], "Hailo TOPS benchmark")
    assert len(contradictions) >= 1
    assert any("TOPS" in c.topic for c in contradictions)


def test_conflicting_sources_remain_preserved():
    """9. Contradiction detection preserves both conflicting sources without discarding either."""
    evaluator = EvidenceEvaluator()
    item_a = EvidenceItem(
        id="ev-a",
        title="Doc A",
        url="https://a.com",
        domain="a.com",
        content="Device supports hardware video encoding.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    item_b = EvidenceItem(
        id="ev-b",
        title="Doc B",
        url="https://b.com",
        domain="b.com",
        content="Device does not support hardware video encoding.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    assessments, gaps, contradictions = evaluator.evaluate([item_a, item_b], "Video encoding")

    # Both items remain assessed
    assert len(assessments) == 2
    assert assessments[0].evidence_id == "ev-a"
    assert assessments[1].evidence_id == "ev-b"
    # Both items marked with contradiction flag
    assert assessments[0].has_contradiction is True
    assert assessments[1].has_contradiction is True
    assert len(contradictions) == 1
