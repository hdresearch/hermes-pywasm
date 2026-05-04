"""
System prompt construction — py2wasm-compatible.

Pure Python string building.  No Jinja2, no file I/O, no YAML.
The host provides memory text, skills data, etc. via the protocol.
"""

from typing import List, Optional


DEFAULT_SYSTEM_PROMPT = """\
You are Hermes, a helpful AI assistant. You are thoughtful, precise, and honest.
When you don't know something, you say so. When a task is ambiguous, you ask
clarifying questions.

You have access to tools that let you accomplish tasks. Use them when appropriate.
Think step by step for complex problems. When making changes, verify your work.

Be concise in your responses unless the user asks for detail."""


def build_system_prompt(
    base_prompt: str = None,
    memory_block: str = "",
    user_block: str = "",
    skills_block: str = "",
    todo_block: str = "",
    extra_context: str = "",
) -> str:
    """Assemble the full system prompt from parts."""
    parts = [base_prompt or DEFAULT_SYSTEM_PROMPT]

    if memory_block:
        parts.append(memory_block)
    if user_block:
        parts.append(user_block)
    if skills_block:
        parts.append(skills_block)
    if todo_block:
        parts.append(todo_block)
    if extra_context:
        parts.append(extra_context)

    return "\n\n".join(parts)


def build_memory_block(memory_text: str) -> str:
    """Format memory entries for system prompt injection."""
    if not memory_text or not memory_text.strip():
        return ""
    return (
        "## Agent Memories\n"
        "These are your persistent notes from previous sessions:\n\n"
        f"{memory_text.strip()}"
    )


def build_skills_block(skills: List[dict]) -> str:
    """Format skills list for system prompt injection.

    Args:
        skills: List of {"name": "...", "description": "...", "content": "..."} dicts.
    """
    if not skills:
        return ""

    lines = ["## Available Skills"]
    lines.append("You have these reusable workflows saved:")
    lines.append("")

    for skill in skills:
        name = skill.get("name", "untitled")
        desc = skill.get("description", "")
        lines.append(f"- **{name}**: {desc}")

    return "\n".join(lines)


def build_todo_block(todo_items: List[dict]) -> str:
    """Format todo items for system prompt injection."""
    if not todo_items:
        return ""

    lines = ["## Current Task List"]

    status_icons = {
        "pending": "⬜",
        "in_progress": "🔄",
        "completed": "✅",
        "cancelled": "❌",
    }

    for item in todo_items:
        icon = status_icons.get(item.get("status", "pending"), "⬜")
        lines.append(f"{icon} [{item.get('id', '?')}] {item.get('content', '')}")

    return "\n".join(lines)
