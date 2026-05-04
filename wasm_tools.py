"""
Tool registry and implementations — py2wasm-compatible.

THREE TIERS of tool execution:
  1. LOCAL  — runs entirely inside the Wasm sandbox (pure Python, no I/O)
  2. INTERACTIVE — host handles user interaction, Wasm parses result
  3. REMOTE — host executes the tool and returns result to Wasm

The host sends the real schemas for all remote tools at init time via the
``host_tool_schemas`` field.  This avoids maintaining duplicate schemas that
can drift from the actual hermes-agent definitions.

Local tools:   memory, todo
Interactive:   clarify
Remote:        whatever the host provides (terminal, read_file, web_search, …)
"""

import json
from typing import Any, Dict, List, Optional


# =========================================================================
# Local + interactive tool schemas (owned by the Wasm binary)
# =========================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Persistent memory that survives across sessions. Two targets:\n"
        "- 'memory': agent's personal notes (environment facts, project conventions)\n"
        "- 'user': what you know about the user (preferences, communication style)\n\n"
        "Actions: add, replace, remove, read"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "read"],
                "description": "The action to perform.",
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store to operate on.",
            },
            "content": {
                "type": "string",
                "description": "Content for add/replace actions.",
            },
            "match": {
                "type": "string",
                "description": "Substring to match for replace/remove actions.",
            },
        },
        "required": ["action", "target"],
    },
}

TODO_SCHEMA = {
    "name": "todo",
    "description": (
        "Manage a task list for the current session. Provide 'todos' to write, "
        "omit to read. Items: {id, content, status}. "
        "Statuses: pending, in_progress, completed, cancelled."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                        },
                    },
                    "required": ["id", "content", "status"],
                },
                "description": "Full replacement todo list. Omit to read current state.",
            },
        },
    },
}

CLARIFY_SCHEMA = {
    "name": "clarify",
    "description": (
        "Ask the user a question when you need clarification. "
        "Supports multiple-choice (up to 4 choices) or open-ended."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to present.",
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
                "description": "Optional answer choices. Omit for open-ended.",
            },
        },
        "required": ["question"],
    },
}


# =========================================================================
# Built-in registry (local + interactive only)
# =========================================================================

_BUILTIN_TOOLS: Dict[str, Dict[str, Any]] = {}

def _register(schema, toolset, locality):
    _BUILTIN_TOOLS[schema["name"]] = {
        "toolset": toolset,
        "schema": schema,
        "locality": locality,
    }

_register(MEMORY_SCHEMA, "memory", "local")
_register(TODO_SCHEMA, "planning", "local")
_register(CLARIFY_SCHEMA, "clarify", "interactive")


# =========================================================================
# Dynamic registry (populated at init from built-ins + host schemas)
# =========================================================================

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
LOCAL_TOOL_NAMES: frozenset = frozenset()
REMOTE_TOOL_NAMES: frozenset = frozenset()
INTERACTIVE_TOOL_NAMES: frozenset = frozenset()
ALL_TOOL_NAMES: frozenset = frozenset()


def init_registry(host_tool_schemas: List[dict] = None):
    """Initialize the full tool registry.

    Called once during agent startup after receiving the init message.

    Args:
        host_tool_schemas: OpenAI-format tool definitions from the host,
            e.g. [{"type": "function", "function": {"name": "terminal", ...}}, ...]
            These are the *real* schemas extracted from the hermes-agent registry.
    """
    global TOOL_REGISTRY, LOCAL_TOOL_NAMES, REMOTE_TOOL_NAMES
    global INTERACTIVE_TOOL_NAMES, ALL_TOOL_NAMES

    TOOL_REGISTRY = dict(_BUILTIN_TOOLS)

    # Merge host-provided schemas
    if host_tool_schemas:
        for tool_def in host_tool_schemas:
            fn = tool_def.get("function", tool_def)
            name = fn.get("name", "")
            if not name:
                continue
            # Skip if we already own this tool (local/interactive)
            if name in TOOL_REGISTRY:
                continue
            TOOL_REGISTRY[name] = {
                "toolset": _infer_toolset(name),
                "schema": fn,
                "locality": "remote",
            }

    LOCAL_TOOL_NAMES = frozenset(
        n for n, info in TOOL_REGISTRY.items() if info["locality"] == "local"
    )
    REMOTE_TOOL_NAMES = frozenset(
        n for n, info in TOOL_REGISTRY.items() if info["locality"] == "remote"
    )
    INTERACTIVE_TOOL_NAMES = frozenset(
        n for n, info in TOOL_REGISTRY.items() if info["locality"] == "interactive"
    )
    ALL_TOOL_NAMES = frozenset(TOOL_REGISTRY.keys())


def _infer_toolset(name: str) -> str:
    """Best-effort toolset inference from tool name."""
    _MAP = {
        "terminal": "terminal", "process": "terminal",
        "read_file": "file", "write_file": "file", "patch": "file", "search_files": "file",
        "web_search": "web", "web_extract": "web",
        "browser_navigate": "browser", "browser_snapshot": "browser",
        "browser_click": "browser", "browser_type": "browser",
        "browser_scroll": "browser", "browser_back": "browser",
        "browser_press": "browser", "browser_close": "browser",
        "browser_get_images": "browser", "browser_vision": "browser",
        "skills_list": "skills", "skill_view": "skills", "skill_manage": "skills",
        "vision_analyze": "vision", "image_generate": "image_gen",
        "execute_code": "code_execution",
        "delegate_task": "delegation", "mixture_of_agents": "moa",
        "session_search": "session_search", "text_to_speech": "tts",
    }
    return _MAP.get(name, "unknown")


# Initialize with builtins only (host schemas added later via init_registry)
init_registry()


# =========================================================================
# Query functions
# =========================================================================

def get_tool_definitions(enabled_toolsets=None, disabled_toolsets=None,
                         host_available_tools=None) -> List[dict]:
    """Return OpenAI-format tool schemas for registered tools.

    Args:
        enabled_toolsets: If set, only include tools from these toolsets.
        disabled_toolsets: If set, exclude tools from these toolsets.
        host_available_tools: If set, only include remote tools the host
            has confirmed it can execute. Local + interactive always included.
    """
    result = []
    for name in sorted(TOOL_REGISTRY):
        info = TOOL_REGISTRY[name]
        toolset = info["toolset"]

        if enabled_toolsets and toolset not in enabled_toolsets:
            continue
        if disabled_toolsets and toolset in disabled_toolsets:
            continue

        if host_available_tools is not None and info["locality"] == "remote":
            if name not in host_available_tools:
                continue

        result.append({"type": "function", "function": info["schema"]})
    return result


def is_local_tool(name: str) -> bool:
    return name in LOCAL_TOOL_NAMES

def is_interactive_tool(name: str) -> bool:
    return name in INTERACTIVE_TOOL_NAMES

def get_tool_locality(name: str) -> str:
    info = TOOL_REGISTRY.get(name)
    return info["locality"] if info else "unknown"


# =========================================================================
# Local tool implementations (pure Python, no I/O)
# =========================================================================

ENTRY_DELIMITER = "\n§\n"


class MemoryStore:
    """In-memory representation of the memory stores."""

    def __init__(self):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = 2200
        self.user_char_limit = 1375

    def load_from_text(self, memory_text: str, user_text: str):
        self.memory_entries = self._parse(memory_text)
        self.user_entries = self._parse(user_text)

    def _parse(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        entries = text.split(ENTRY_DELIMITER)
        return [e.strip() for e in entries if e.strip()]

    def _render(self, entries: List[str]) -> str:
        return ENTRY_DELIMITER.join(entries)

    def _entries_for(self, target: str) -> List[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        return len(ENTRY_DELIMITER.join(entries)) if entries else 0

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def handle(self, args: dict) -> dict:
        action = args.get("action", "read")
        target = args.get("target", "memory")
        content = args.get("content", "").strip()
        match_str = args.get("match", "").strip()

        if target not in ("memory", "user"):
            return {"success": False, "error": f"Invalid target: {target}"}

        if action == "read":
            return self._read(target)
        elif action == "add":
            return self._add(target, content)
        elif action == "replace":
            return self._replace(target, match_str, content)
        elif action == "remove":
            return self._remove(target, match_str)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    def _read(self, target):
        entries = self._entries_for(target)
        return {
            "success": True, "target": target, "entries": entries,
            "count": len(entries), "chars": self._char_count(target),
            "limit": self._char_limit(target),
        }

    def _add(self, target, content):
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        entries = self._entries_for(target)
        limit = self._char_limit(target)
        if content in entries:
            return {"success": True, "message": "Entry already exists."}
        new_entries = entries + [content]
        new_total = len(ENTRY_DELIMITER.join(new_entries))
        if new_total > limit:
            return {"success": False, "error": f"Would exceed {target} limit ({new_total}/{limit} chars)."}
        self._set_entries(target, new_entries)
        return {"success": True, "target": target, "action": "add",
                "entries": new_entries, "count": len(new_entries),
                "chars": len(ENTRY_DELIMITER.join(new_entries)), "limit": limit}

    def _replace(self, target, match_str, content):
        if not match_str:
            return {"success": False, "error": "match is required for replace."}
        if not content:
            return {"success": False, "error": "content is required for replace."}
        entries = self._entries_for(target)
        matches = [i for i, e in enumerate(entries) if match_str.lower() in e.lower()]
        if not matches:
            return {"success": False, "error": f"No entry matches '{match_str}'."}
        if len(matches) > 1:
            return {"success": False, "error": f"Multiple entries match '{match_str}'."}
        idx = matches[0]
        new_entries = list(entries)
        new_entries[idx] = content
        limit = self._char_limit(target)
        new_total = len(ENTRY_DELIMITER.join(new_entries))
        if new_total > limit:
            return {"success": False, "error": f"Would exceed limit ({new_total}/{limit} chars)."}
        self._set_entries(target, new_entries)
        return {"success": True, "target": target, "action": "replace",
                "entries": new_entries, "count": len(new_entries),
                "chars": new_total, "limit": limit}

    def _remove(self, target, match_str):
        if not match_str:
            return {"success": False, "error": "match is required for remove."}
        entries = self._entries_for(target)
        matches = [i for i, e in enumerate(entries) if match_str.lower() in e.lower()]
        if not matches:
            return {"success": False, "error": f"No entry matches '{match_str}'."}
        if len(matches) > 1:
            return {"success": False, "error": f"Multiple entries match '{match_str}'."}
        new_entries = [e for i, e in enumerate(entries) if i != matches[0]]
        self._set_entries(target, new_entries)
        return {"success": True, "target": target, "action": "remove",
                "entries": new_entries, "count": len(new_entries),
                "chars": self._char_count(target), "limit": self._char_limit(target)}


_memory_store = MemoryStore()


VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

def _handle_todo(args: dict, todo_items: List[dict]) -> dict:
    new_todos = args.get("todos")
    if new_todos is None:
        return {"todos": todo_items, "count": len(todo_items)}
    validated = []
    for t in new_todos:
        item = {
            "id": str(t.get("id", "")).strip(),
            "content": str(t.get("content", "")).strip(),
            "status": str(t.get("status", "pending")).strip(),
        }
        if not item["id"] or not item["content"]:
            continue
        if item["status"] not in VALID_STATUSES:
            item["status"] = "pending"
        validated.append(item)
    return {"todos": validated, "count": len(validated)}


# =========================================================================
# Dispatch
# =========================================================================

def dispatch_local_tool(
    name: str,
    args: dict,
    todo_items: List[dict] = None,
    memory_text: str = "",
    user_text: str = "",
) -> str:
    """Execute a local tool and return JSON result string."""
    try:
        if name == "memory":
            _memory_store.load_from_text(memory_text, user_text)
            result = _memory_store.handle(args)
            return json.dumps(result, ensure_ascii=False)
        elif name == "todo":
            result = _handle_todo(args, todo_items or [])
            return json.dumps(result, ensure_ascii=False)
        else:
            return json.dumps({"error": f"Unknown local tool: {name}"})
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {type(e).__name__}: {e}"})
