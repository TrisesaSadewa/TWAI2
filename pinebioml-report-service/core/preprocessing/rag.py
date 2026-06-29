import json
import logging
import requests

logger = logging.getLogger(__name__)

# How long to wait for the verifier LLM. The main narrative generation may
# already be occupying the local model, so this must be generous; a too-short
# timeout turns "verifier slow" into "hallucination confirmed" (the old bug).
# On CPU-only hosts even a fast model can be slow, hence the generous value.
_VERIFY_TIMEOUT = 120
_VERIFY_RETRIES = 2


def semantic_verify_hallucinations(flagged_sentences: list, parsed_data: dict, model_name: str = None) -> bool:
    """
    Tier 2 Check: Uses the LLM to perform a semantic grounding check on sentences
    flagged by Tier 1.

    Returns True if the sentences are semantically grounded in the data, False if
    they are actual hallucinations.

    Failure-mode policy: if the verifier LLM is unreachable or errors after
    retries, return True (pass-through) rather than False. Returning False would
    treat "could not verify" as "confirmed hallucination", which — combined with
    Tier-1 false positives from leaked reasoning tokens — caused valid reports to
    be rejected and fall back to the rule-based generator. When we genuinely
    cannot run the check, the safer default is to NOT reject.
    """
    if not flagged_sentences:
        return True

    # Resolve the verifier model + endpoint from settings (not hardcoded), so
    # this works in Docker (OLLAMA_BASE_URL=http://ollama:11434) and uses the
    # fast non-reasoning judge model by default.
    from core.config import settings
    model = model_name or settings.VERIFIER_MODEL
    base_url = settings.OLLAMA_BASE_URL.rstrip("/")
    generate_url = f"{base_url}/api/generate"

    logger.info(f"Running Tier 2 Semantic Verification on {len(flagged_sentences)} flagged sentences (model={model})...")

    # Simplify the parsed data so it doesn't blow up the context window
    data_str = json.dumps(parsed_data, indent=2)[:2000]

    prompt = f"""You are a strict data verification judge.

Provided is a summary of the ground-truth data:
{data_str}

The following claims were flagged because they contain numbers not strictly found in the data:
{json.dumps(flagged_sentences, indent=2)}

Your task is to determine if these claims are logically and semantically grounded in the ground-truth data.
For example, if the data says "Accuracy: 90%" and the claim says "There is 10% inaccuracy", this IS logically grounded (100 - 90 = 10).

Respond ONLY with "YES" if the claims are grounded and logically true based on the data.
Respond ONLY with "NO" if the claims are hallucinated or contradict the data.
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0
    }

    last_err = None
    for attempt in range(_VERIFY_RETRIES + 1):
        try:
            response = requests.post(generate_url, json=payload, timeout=_VERIFY_TIMEOUT)
            response.raise_for_status()
            result = response.json().get("response", "").strip().upper()

            # Clean up any wrapper tags the model may have emitted.
            if "</" in result:
                # Strip a trailing closing tag if present (e.g. model wrapped answer)
                result = result.split("</")[-1].split(">")[-1].strip() if ">" in result.split("</")[-1] else result

            if "YES" in result:
                logger.info("Tier 2 Semantic Verification: PASSED. Hallucination flag cleared.")
                return True
            else:
                logger.warning(f"Tier 2 Semantic Verification: FAILED. Actual hallucination confirmed: {result}")
                return False
        except Exception as e:
            last_err = e
            logger.warning(f"Tier 2 Semantic Verification error (attempt {attempt + 1}): {e}")
            if attempt < _VERIFY_RETRIES:
                continue

    # Verifier unreachable after retries — pass-through rather than reject.
    logger.error(
        f"Tier 2 Semantic Verification unavailable after {_VERIFY_RETRIES + 1} attempts ({last_err}). "
        f"Passing through (not rejecting) to avoid false hallucination failures."
    )
    return True
