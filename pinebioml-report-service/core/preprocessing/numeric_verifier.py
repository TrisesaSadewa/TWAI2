import re
from typing import Dict, Any, List, Set

def extract_numbers(text: str) -> List[float]:
    """
    Extracts all numbers (integers, floats, percentages) from a given string.
    """
    # Matches numbers like 95, 0.95, .95, 95.5
    pattern = r'\b\d*\.?\d+\b'
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            pass
    return numbers

def get_all_valid_numbers(parsed_data: Dict[str, Any]) -> Set[float]:
    """
    Recursively extracts all valid numbers from the original parsed CSV/Image data.
    This acts as our "ground truth" to prevent hallucinations.
    """
    valid_numbers = set()
    
    def traverse(item):
        if isinstance(item, (int, float)):
            valid_numbers.add(float(item))
            # Also add common percentage representations (e.g., 0.95 -> 95.0)
            if 0 <= item <= 1:
                valid_numbers.add(float(round(item * 100, 2)))
                valid_numbers.add(float(round(item * 100, 1)))
                valid_numbers.add(float(int(item * 100)))
        elif isinstance(item, dict):
            for v in item.values():
                traverse(v)
        elif isinstance(item, list):
            for v in item:
                traverse(v)
        elif isinstance(item, str):
            # Sometimes numbers are stored as strings in parsed data
            try:
                val = float(item)
                valid_numbers.add(val)
                if 0 <= val <= 1:
                    valid_numbers.add(float(round(val * 100, 2)))
                    valid_numbers.add(float(round(val * 100, 1)))
                    valid_numbers.add(float(int(val * 100)))
            except ValueError:
                pass
                
    traverse(parsed_data)
    return valid_numbers

def verify_no_hallucinations(generated_json: Dict[str, Any], parsed_data: Dict[str, Any]) -> List[str]:
    """
    Tier 1 Check: Cross-checks all numerical values in the LLM-generated JSON against the original parsed data.
    Returns a list of sentences/fields that contain ungrounded numbers.
    """
    valid_numbers = get_all_valid_numbers(parsed_data)
    acceptable_constants = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0}
    
    flagged_sentences = []
    
    for key, value in generated_json.items():
        if isinstance(value, str):
            extracted_numbers = extract_numbers(value)
            for num in extracted_numbers:
                if num in acceptable_constants:
                    continue
                matched = any(abs(num - v) < 0.01 for v in valid_numbers)
                if not matched:
                    flagged_sentences.append(value)
                    break # One hallucination in this field is enough to flag the sentence
                    
    if flagged_sentences:
        print(f"WARNING: Tier 1 flagged {len(flagged_sentences)} fields for potential hallucinations.")
        
    return flagged_sentences
