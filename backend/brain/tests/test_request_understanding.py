import pytest
from datetime import datetime, timezone
import dataclasses

from core.interfaces.request_understanding_interface import RequestUnderstandingInterface
from core.models.request import Request
from brain.request_understanding import (
    StandardRequestUnderstanding,
    RequestUnderstandingError,
    normalize_text,
)


@pytest.fixture
def parser():
    return StandardRequestUnderstanding()


# ============================================================================
# INTERFACE & BASELINE TESTS
# ============================================================================

def test_interface_compliance(parser):
    """1. StandardRequestUnderstanding must implement RequestUnderstandingInterface."""
    assert isinstance(parser, RequestUnderstandingInterface)
    assert hasattr(parser, "understand")
    assert callable(parser.understand)


def test_raw_string_input(parser):
    """2. Understands a raw string input and produces a valid Request."""
    req = parser.understand("Hello, please summarize the documentation.")
    assert isinstance(req, Request)
    assert req.original_text == "Hello, please summarize the documentation."
    assert req.normalized_text == "Hello, please summarize the documentation."
    assert req.source == "chat"
    assert req.session_id == "default"
    assert isinstance(req.parameters, dict)
    assert isinstance(req.constraints, dict)


def test_original_text_exact_preservation(parser):
    """3. original_text is preserved EXACTLY as supplied (whitespace, formatting, newlines)."""
    raw = "   \n\t  Important: Keep This Exact Spacing!   \t \n"
    req = parser.understand(raw)
    assert req.original_text == raw
    assert req.original_text != req.normalized_text


def test_nfkc_normalization():
    """4. Unicode NFKC normalization converts full-width and compatibility characters."""
    # Fullwidth latin characters and ligature
    raw = "Ｈｅｌｌｏ ﬁle"
    normalized = normalize_text(raw)
    assert normalized == "Hello file"


def test_whitespace_normalization():
    """5. Normalizes repeated whitespace, tabs, and newlines into single spaces."""
    raw = "Search   \t\t\n   in    multiple     locations   \n\n"
    normalized = normalize_text(raw)
    assert normalized == "Search in multiple locations"


def test_casing_preservation():
    """6. Meaningful casing is strictly preserved without blanket lowercasing."""
    raw = "Load C:\\Projects\\AI-Control-Center\\README.md with PyTorch"
    normalized = normalize_text(raw)
    assert normalized == "Load C:\\Projects\\AI-Control-Center\\README.md with PyTorch"
    assert "README.md" in normalized
    assert "PyTorch" in normalized
    assert "C:\\Projects" in normalized


# ============================================================================
# DICTIONARY PAYLOAD TESTS
# ============================================================================

def test_dictionary_payload_basic(parser):
    """7. Understands a structured dictionary payload with text field."""
    payload = {
        "text": "Display network metrics",
        "source": "voice",
        "session_id": "sess-voice-123",
    }
    req = parser.understand(payload)
    assert req.original_text == "Display network metrics"
    assert req.normalized_text == "Display network metrics"
    assert req.source == "voice"
    assert req.session_id == "sess-voice-123"


def test_dictionary_payload_with_original_text_key(parser):
    """8. Understands dictionary payload providing 'original_text' instead of 'text'."""
    payload = {
        "original_text": "Run diagnostic check",
        "source": "api",
    }
    req = parser.understand(payload)
    assert req.original_text == "Run diagnostic check"
    assert req.source == "api"


def test_dictionary_payload_defaults(parser):
    """9. Dictionary payload without optional fields applies correct defaults."""
    payload = {"text": "Simple query"}
    req = parser.understand(payload)
    assert req.source == "chat"
    assert req.session_id == "default"


def test_explicit_parameters_preserved(parser):
    """10. Explicitly supplied parameters in payload are preserved."""
    payload = {
        "text": "Retrieve system architecture",
        "parameters": {"collection": "engineering_docs", "author": "Alice"},
    }
    req = parser.understand(payload)
    assert req.parameters["collection"] == "engineering_docs"
    assert req.parameters["author"] == "Alice"


def test_explicit_constraints_preserved(parser):
    """11. Explicitly supplied constraints in payload are preserved."""
    payload = {
        "text": "Find user guides",
        "constraints": {"max_results": 15, "timeout_seconds": 25.0, "privacy_level": "local_only"},
    }
    req = parser.understand(payload)
    assert req.constraints["max_results"] == 15
    assert req.constraints["timeout_seconds"] == 25.0
    assert req.constraints["privacy_level"] == "local_only"


# ============================================================================
# SIMPLE DETERMINISTIC EXTRACTION TESTS
# ============================================================================

def test_simple_path_extraction(parser):
    """12. Obvious file/directory path is extracted into parameters if not provided."""
    req = parser.understand("Inspect the log file at C:\\logs\\app.log please")
    assert "path" in req.parameters
    assert req.parameters["path"] == "C:\\logs\\app.log"


def test_simple_query_extraction(parser):
    """13. Obvious search query is extracted into parameters."""
    req = parser.understand("Please search for 'kernel initialization' in docs")
    assert req.parameters.get("query") == "kernel initialization"


def test_simple_constraint_extraction_limit(parser):
    """14. Simple 'limit' or 'top' constraint is extracted into constraints."""
    req = parser.understand("Retrieve articles with limit 10")
    assert req.constraints.get("max_results") == 10


def test_simple_constraint_extraction_timeout(parser):
    """15. Simple 'timeout' constraint is extracted into constraints."""
    req = parser.understand("Run diagnostic timeout 30s")
    assert req.constraints.get("timeout_seconds") == 30.0


def test_simple_constraint_extraction_privacy_and_confirmation(parser):
    """16. Privacy and confirmation flags are extracted into constraints."""
    req = parser.understand("Delete temporary files local only confirm first")
    assert req.constraints.get("privacy_level") == "local_only"
    assert req.constraints.get("confirmation_required") is True


# ============================================================================
# EMPTY & AMBIGUOUS INPUT TESTS
# ============================================================================

def test_empty_string_input(parser):
    """17. Empty string input returns valid Request with empty quality flags."""
    req = parser.understand("")
    assert req.original_text == ""
    assert req.normalized_text == ""
    assert req.parameters["is_empty"] is True
    assert req.parameters["valid"] is False
    assert req.parameters["is_ambiguous"] is True
    assert req.parameters["clarification_needed"] is True


def test_whitespace_only_string_input(parser):
    """18. Whitespace-only input returns valid Request with empty quality flags."""
    req = parser.understand("   \t  \n  ")
    assert req.normalized_text == ""
    assert req.parameters["is_empty"] is True
    assert req.parameters["valid"] is False


def test_ambiguous_punctuation_only_input(parser):
    """19. Input with only punctuation is flagged as ambiguous."""
    req = parser.understand("???")
    assert req.parameters["is_empty"] is False
    assert req.parameters["is_ambiguous"] is True
    assert req.parameters["clarification_needed"] is True
    assert "punctuation" in req.parameters["ambiguity_reason"].lower()


def test_incomplete_query_input(parser):
    """20. Incomplete command is flagged as ambiguous without crashing."""
    req = parser.understand("search for")
    assert req.parameters["is_ambiguous"] is True
    assert req.parameters["clarification_needed"] is True
    assert "incomplete" in req.parameters["ambiguity_reason"].lower()


def test_deictic_vague_input(parser):
    """21. Vague deictic inputs like 'do it' are flagged as ambiguous."""
    req = parser.understand("do it")
    assert req.parameters["is_ambiguous"] is True
    assert req.parameters["clarification_needed"] is True


# ============================================================================
# IMMUTABILITY, METADATA & ERROR HANDLING TESTS
# ============================================================================

def test_uuid_uniqueness_and_format(parser):
    """22. Request IDs are generated with unique UUIDs."""
    req1 = parser.understand("test 1")
    req2 = parser.understand("test 2")
    assert req1.id != req2.id
    assert req1.id.startswith("req-")
    assert req2.id.startswith("req-")


def test_timestamp_validity(parser):
    """23. Timestamp is a valid datetime recorded at call time."""
    before = datetime.now()
    req = parser.understand("timestamp test")
    after = datetime.now()
    assert isinstance(req.timestamp, datetime)
    assert before <= req.timestamp <= after


def test_request_immutability(parser):
    """24. Returned Request instance is frozen and immutable."""
    req = parser.understand("immutable request")
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.original_text = "modified"  # type: ignore
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.source = "voice"  # type: ignore


def test_malformed_input_type_error(parser):
    """25. Malformed non-str, non-dict input raises RequestUnderstandingError."""
    with pytest.raises(RequestUnderstandingError) as excinfo:
        parser.understand([1, 2, 3])  # type: ignore
    assert "Unsupported input type" in str(excinfo.value)


def test_malformed_dict_missing_text_error(parser):
    """26. Dictionary payload missing text raises RequestUnderstandingError."""
    with pytest.raises(RequestUnderstandingError) as excinfo:
        parser.understand({"source": "api"})
    assert "must contain 'text' or 'original_text'" in str(excinfo.value)


def test_malformed_dict_invalid_parameters_error(parser):
    """27. Dictionary payload with non-dict parameters raises RequestUnderstandingError."""
    with pytest.raises(RequestUnderstandingError) as excinfo:
        parser.understand({"text": "test", "parameters": "not-a-dict"})
    assert "Payload 'parameters' must be a dictionary" in str(excinfo.value)
