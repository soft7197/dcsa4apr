"""Robust JSON parsing for LLM outputs that may contain non-JSON text."""

import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_json(text: str, fallback: dict = None) -> dict:
    """
    Extract JSON from LLM output that may contain markdown, extra text, etc.

    Tries multiple strategies:
    1. Direct json.loads
    2. Extract from ```json ... ``` code blocks
    3. Find the outermost { ... } or [ ... ] in the text
    4. Return fallback if all fail

    Args:
        text: Raw LLM output string.
        fallback: Value to return if parsing fails. Defaults to empty dict.

    Returns:
        Parsed JSON as a dict (or list).
    """
    if fallback is None:
        fallback = {}

    if not text or not text.strip():
        logger.warning("Empty LLM response, returning fallback")
        return fallback

    text = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find outermost JSON object or array using bracket matching
    for open_char, close_char in [('{', '}'), ('[', ']')]:
        start = text.find(open_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        # If bracket matching didn't work, continue to next bracket type

    logger.warning("Could not parse JSON from LLM response, returning fallback")
    logger.debug(f"Failed response (first 500 chars): {text[:500]}")
    return fallback
