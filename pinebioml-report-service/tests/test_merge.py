import json
from core.report.narrative_generator import NarrativeGenerator

gen = NarrativeGenerator()

llm_output = {
    "expert": {
        "context_analysis": "This is context",
        "executive_summary": "Exec sum",
        "conclusion": "Wait"
    }
}
fallback = {
    "expert": {
        "conclusion": "Rule conclusion",
        "recommendations": "Rule recommendations"
    }
}
bad_sections = ["expert.conclusion", "expert.recommendations"]

merged = gen._merge_with_fallback(llm_output, fallback, bad_sections)
print(json.dumps(merged, indent=2))
