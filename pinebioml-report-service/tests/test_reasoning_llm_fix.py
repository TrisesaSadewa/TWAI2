import pytest
from core.report.narrative_generator import (
    _extract_llm_response_text,
    _extract_llm_chunk_text,
    _clean_and_parse_llm_json,
    NarrativeGenerator
)


# ── _extract_llm_response_text: tuple API ────────────────────────────────────

def test_extract_llm_response_text_returns_tuple():
    """Verify return type is a (content, reasoning) tuple."""
    payload = {
        "choices": [{
            "message": {
                "content": '{"expert": {}}',
                "reasoning": "Some thinking process"
            }
        }]
    }
    result = _extract_llm_response_text(payload)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_extract_llm_response_text_separates_content_and_reasoning():
    """Content and reasoning must be returned in separate tuple elements."""
    payload = {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '{"expert": {"findings": "AUC = 0.98"}}',
                "reasoning": "Thinking Process:\n1. Analyze the data\n2. Write report"
            },
            "finish_reason": "stop"
        }]
    }
    content, reasoning = _extract_llm_response_text(payload)
    assert '{"expert"' in content
    assert "Thinking Process" in reasoning


def test_extract_llm_response_text_empty_content_reasoning_only():
    """When content is empty, reasoning should still be returned separately."""
    payload = {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning": 'Thinking Process:\n1. Analyze data\n```json\n{"recommendations": "Suggested steps"}\n```'
            },
            "finish_reason": "length"
        }]
    }
    content, reasoning = _extract_llm_response_text(payload)
    assert content == ""
    assert "Thinking Process" in reasoning


def test_extract_llm_response_text_content_only_no_reasoning():
    """Non-reasoning models return content with empty reasoning."""
    payload = {
        "choices": [{
            "message": {
                "content": '{"expert": {"findings": "Good model"}}',
            }
        }]
    }
    content, reasoning = _extract_llm_response_text(payload)
    assert '{"expert"' in content
    assert reasoning == ""


def test_extract_llm_response_text_ollama_native_with_reasoning():
    """Ollama native chat format with separate content/reasoning fields."""
    payload = {
        "message": {
            "content": '{"expert": {"summary": "Test"}}',
            "reasoning_content": "I need to analyze this carefully"
        }
    }
    content, reasoning = _extract_llm_response_text(payload)
    assert '{"expert"' in content
    assert "analyze this carefully" in reasoning


def test_extract_llm_response_text_empty_payload():
    """Empty/invalid payloads should return empty tuple."""
    assert _extract_llm_response_text({}) == ("", "")
    assert _extract_llm_response_text(None) == ("", "")
    assert _extract_llm_response_text("not a dict") == ("", "")


# ── _extract_llm_chunk_text: tuple API ───────────────────────────────────────

def test_extract_llm_chunk_text_returns_tuple():
    """Verify return type is a (content, reasoning) tuple."""
    chunk = {
        "choices": [{
            "delta": {
                "content": "partial json",
                "reasoning_content": "thinking"
            }
        }]
    }
    result = _extract_llm_chunk_text(chunk)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_extract_llm_chunk_text_separates_content_and_reasoning():
    """Content and reasoning from streaming delta must be separated."""
    chunk = {
        "choices": [{
            "delta": {
                "content": '{"expert":',
                "reasoning_content": "Step 1: analyze"
            }
        }]
    }
    content, reasoning = _extract_llm_chunk_text(chunk)
    assert content == '{"expert":'
    assert reasoning == "Step 1: analyze"


def test_extract_llm_chunk_text_reasoning_only_no_content():
    """Reasoning-only chunks (during thinking phase) must NOT leak into content."""
    chunk = {
        "choices": [{
            "delta": {
                "content": "",
                "reasoning": "Analyzing the request carefully"
            }
        }]
    }
    content, reasoning = _extract_llm_chunk_text(chunk)
    assert content == ""
    assert reasoning == "Analyzing the request carefully"


def test_extract_llm_chunk_text_content_only_no_reasoning():
    """Content-only chunks (during output phase) have empty reasoning."""
    chunk = {
        "choices": [{
            "delta": {
                "content": '"findings": "AUC=0.98"'
            }
        }]
    }
    content, reasoning = _extract_llm_chunk_text(chunk)
    assert content == '"findings": "AUC=0.98"'
    assert reasoning == ""


def test_extract_llm_chunk_text_empty_chunk():
    """Empty chunks should return empty tuple."""
    assert _extract_llm_chunk_text({}) == ("", "")
    assert _extract_llm_chunk_text(None) == ("", "")


# ── Streaming buffer separation simulation ───────────────────────────────────

def test_streaming_buffer_separation():
    """Simulate a full streaming session with thinking phase followed by content phase.
    
    This is the core scenario: qwen3.5:9b sends reasoning-only chunks first,
    then content-only chunks. The old code mixed them together; the new code
    must keep them separate.
    """
    # Simulate the chunk sequence a reasoning model would produce
    chunks = [
        # Thinking phase: content is empty, reasoning has thinking text
        {"choices": [{"delta": {"content": "", "reasoning_content": "Thinking Process:\n"}}]},
        {"choices": [{"delta": {"content": "", "reasoning_content": "1. Analyze the data\n"}}]},
        {"choices": [{"delta": {"content": "", "reasoning_content": "2. Write the report\n"}}]},
        # Output phase: content has JSON, reasoning is empty
        {"choices": [{"delta": {"content": '{"expert": {'}}]},
        {"choices": [{"delta": {"content": '"findings": "AUC of 0.98"'}}]},
        {"choices": [{"delta": {"content": "}}"}}]},
    ]
    
    content_buf = ""
    reasoning_buf = ""
    for chunk in chunks:
        c, r = _extract_llm_chunk_text(chunk)
        if c:
            content_buf += c
        if r:
            reasoning_buf += r
    
    # Content buffer must contain ONLY the JSON, not reasoning text
    assert content_buf == '{"expert": {"findings": "AUC of 0.98"}}'
    assert "Thinking Process" not in content_buf
    
    # Reasoning buffer must contain ONLY the thinking text
    assert "Thinking Process" in reasoning_buf
    assert '{"expert"' not in reasoning_buf


# ── _clean_and_parse_llm_json ────────────────────────────────────────────────

def test_clean_and_parse_llm_json_handles_unclosed_think_tag():
    """Verify that _clean_and_parse_llm_json extracts JSON even if unclosed <think> tag is at start."""
    raw_text = '<think>\nThinking process...\n```json\n{\n  "expert": {\n    "recommendations": "Perform cross validation."\n  }\n}\n```'
    parsed = _clean_and_parse_llm_json(raw_text)
    
    assert isinstance(parsed, dict)
    assert "expert" in parsed
    assert parsed["expert"]["recommendations"] == "Perform cross validation."

def test_clean_and_parse_llm_json_handles_plain_text_thinking_process_prefix():
    """Verify that _clean_and_parse_llm_json ignores plain-text 'Thinking Process:' and informal braces."""
    raw_text = (
        'Thinking Process:\n\n'
        '1. **Analyze the Request:**\n'
        '   * **Role:** Clinical Data Scientist & Biostatistician at PineBioML.\n'
        '   * **Schema:** Specific keys: `expert` (executive_summary, findings, conclusion)\n\n'
        '```json\n'
        '{\n'
        '  "expert": {\n'
        '    "executive_summary": "Model ready for preliminary screening.",\n'
        '    "findings": "Accuracy of 0.9211.",\n'
        '    "recommendations": "Validate on external cohort."\n'
        '  }\n'
        '}\n'
        '```'
    )
    parsed = _clean_and_parse_llm_json(raw_text)
    assert isinstance(parsed, dict)
    assert "expert" in parsed
    assert parsed["expert"]["executive_summary"] == "Model ready for preliminary screening."

def test_clean_and_parse_llm_json_plain_text_section_fallback():
    """Verify that _clean_and_parse_llm_json recovers narrative sections from plain-text reasoning drafts without JSON braces."""
    raw_text = (
        'Thinking Process:\n\n'
        '1. Analyze the Request: Clinical data scientist role.\n'
        '2. Drafting Content:\n'
        '   * Executive Summary: Model achieves an AUC of 0.987 and is ready for preliminary screening.\n'
        '   * Preprocessing: Stratified 80/20 train/test split with SMOTE class balancing.\n'
        '   * Findings: High sensitivity across both classes with mean accuracy 0.9211.\n'
        '   * Recommendations: Perform prospective validation and monitor false positive rate.'
    )
    parsed = _clean_and_parse_llm_json(raw_text)
    assert isinstance(parsed, dict)
    assert "expert" in parsed
    assert "executive_summary" in parsed["expert"]
    assert "AUC of 0.987" in parsed["expert"]["executive_summary"]
    assert "findings" in parsed["expert"]
    assert "mean accuracy 0.9211" in parsed["expert"]["findings"]

def test_clean_and_parse_llm_json_exact_user_log_trace():
    """Test the exact raw text structure from the user's error log where 'Thinking Process:' starts at char 0."""
    raw_text = (
        "Thinking Process:\n\n"
        "1.  **Analyze the Request:**\n"
        "    *   **Role:** Clinical Data Scientist & Biostatistician at PineBioML.\n"
        "    *   **Task:** Generate a comprehensive clinical narrative report based on provided training run data (Breast Cancer Wisconsin dataset).\n"
        "    *   **Output Format:** Valid JSON object ONLY. No markdown code fences, no extra text.\n"
        "    *   **Schema:** Specific keys: `expert` (containing `executive_summary`, `preprocessing_and_data_quality`, `findings`, `conclusion`, `recommendations`).\n\n"
        "2.  **Drafting Content:**\n"
        "    *   Executive Summary: Model achieves AUC-ROC 0.987 and is conditionally suitable for screening.\n"
        "    *   Preprocessing and Data Quality: Stratified 80/20 train/test split utilized.\n"
        "    *   Findings: High sensitivity achieved across all metrics.\n"
        "    *   Recommendations: Conduct prospective validation."
    )
    parsed = _clean_and_parse_llm_json(raw_text)
    assert isinstance(parsed, dict)
    assert "expert" in parsed
    assert "executive_summary" in parsed["expert"]
    assert "0.987" in parsed["expert"]["executive_summary"]
    assert "preprocessing_and_data_quality" in parsed["expert"]

def test_retry_missing_sections_handles_data_anchored_suffix():
    """Verify that _retry_missing_sections cleans 'expert.findings (not data-anchored)' and merges alias keys."""
    gen = NarrativeGenerator()
    missing = [
        "expert.executive_summary",
        "expert.preprocessing_and_data_quality",
        "expert.findings (not data-anchored)"
    ]
    current_json = {"expert": {"conclusion": "Model is stable."}}
    
    # Mock requests.post to return a valid JSON patch
    class DummyResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": '{"executive_summary": "Substantive summary text for patient screening.", "preprocessing_and_data_quality": "Detailed data quality description text.", "findings": "Model performance findings text with precision metrics."}'
                    }
                }]
            }
            
    import unittest.mock as mock
    with mock.patch("requests.post", return_value=DummyResponse()):
        patched = gen._retry_missing_sections(
            missing_sections=missing,
            current_json=current_json,
            data_block="AUC_ROC = 0.98",
            model="qwen3.5:9b",
            headers={},
            url="http://localhost:11434/v1/chat/completions",
            timeout=30,
            model_cfg={"max_tokens": 8192}
        )
        
    assert patched is not None
    assert "expert" in patched
    assert "executive_summary" in patched["expert"]
    assert "findings" in patched["expert"]
    assert "preprocessing_and_data_quality" in patched["expert"]


def test_retry_missing_sections_reasoning_model_content_preferred():
    """Verify targeted retry prefers content over reasoning for reasoning models."""
    gen = NarrativeGenerator()
    missing = ["expert.recommendations"]
    current_json = {"expert": {"findings": "Some findings"}}
    
    class DummyResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": '{"recommendations": "**DATA QUALITY IMPROVEMENTS:** Increase sample diversity. **VALIDATION PROTOCOL:** Use k-fold cross validation."}',
                        "reasoning": "I need to generate recommendations that are specific to this dataset..."
                    }
                }]
            }
    
    import unittest.mock as mock
    with mock.patch("requests.post", return_value=DummyResponse()):
        patched = gen._retry_missing_sections(
            missing_sections=missing,
            current_json=current_json,
            data_block="AUC_ROC = 0.98",
            model="qwen3.5:9b",
            headers={},
            url="http://localhost:11434/v1/chat/completions",
            timeout=30,
            model_cfg={"max_tokens": 8192}
        )
    
    assert patched is not None
    assert "recommendations" in patched["expert"]
    assert "DATA QUALITY IMPROVEMENTS" in patched["expert"]["recommendations"]
    # Reasoning text must NOT leak into the recommendations
    assert "I need to generate" not in patched["expert"]["recommendations"]
