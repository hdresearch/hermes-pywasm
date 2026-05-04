"""
Context management utilities — py2wasm-compatible.

Token estimation, compression decisions, and message history management.
All pure Python stdlib.
"""

from typing import List, Dict, Any, Optional, Tuple


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: List[dict]) -> int:
    """Estimate total tokens across a list of messages.

    Accounts for role/name overhead (~4 tokens per message).
    """
    total = 0
    for msg in messages:
        # Message overhead
        total += 4

        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Multi-part content (vision, etc.)
            for part in content:
                if isinstance(part, dict):
                    total += estimate_tokens(str(part.get("text", "")))
                else:
                    total += estimate_tokens(str(part))

        # Tool calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                total += estimate_tokens(fn.get("name", ""))
                total += estimate_tokens(fn.get("arguments", ""))

    return total


def should_compress(
    messages: List[dict],
    context_length: int,
    threshold_ratio: float = 0.85,
    min_messages: int = 6,
) -> bool:
    """Check if context compression should be triggered."""
    if len(messages) < min_messages:
        return False

    total_tokens = estimate_messages_tokens(messages)
    threshold = int(context_length * threshold_ratio)
    return total_tokens >= threshold


def get_messages_to_compress(
    messages: List[dict],
    protect_first: int = 2,
    protect_last: int = 2,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split messages into (to_compress, keep_before, keep_after).

    Protects the first N and last N messages from compression.
    Returns (middle_messages, first_messages, last_messages).
    """
    if len(messages) <= protect_first + protect_last:
        return [], messages, []

    before = messages[:protect_first]
    after = messages[-protect_last:]
    middle = messages[protect_first:-protect_last] if protect_last > 0 else messages[protect_first:]

    return middle, before, after


def apply_compression_result(
    messages: List[dict],
    keep_before: List[dict],
    keep_after: List[dict],
    summary: str,
) -> List[dict]:
    """Replace compressed middle section with a summary message."""
    summary_msg = {
        "role": "user",
        "content": f"[Previous conversation summary: {summary}]",
    }

    return keep_before + [summary_msg] + keep_after
