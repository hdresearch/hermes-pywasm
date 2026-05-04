"""
WASI host-call protocol for hermes-agent.

Since py2wasm targets WASI (no networking, no threads, no C extensions),
the compiled Wasm binary communicates with a host process via JSON-lines
over stdin/stdout.

The Wasm binary handles:
  - Conversation loop logic (iterations, retry, context management)
  - System prompt building (from config + memories + skills)
  - Tool call parsing and dispatch (for LOCAL tools only)
  - Context compression decisions (when to compress, what to keep)
  - Token estimation heuristics
  - Message history management

The host handles:
  - LLM API calls (HTTP to OpenAI/OpenRouter/etc.)
  - User I/O (terminal, messaging platforms)
  - File I/O outside the WASI sandbox
  - subprocess-dependent tools (terminal, browser, code execution)

Protocol: newline-delimited JSON on stdin (host→wasm) and stdout (wasm→host).
Each message has a "type" field.  Stderr is used for logging.
"""

import json
import sys
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Message Types: Host → Wasm (stdin)
# ---------------------------------------------------------------------------

# { "type": "init", "config": { "model": "...", "api_key": "...", ... } }
# { "type": "user_message", "content": "...", "conversation_history": [...] }
# { "type": "api_response", "response": { ...OpenAI ChatCompletion... } }
# { "type": "tool_result", "tool_call_id": "...", "name": "...", "content": "..." }
# { "type": "clarify_response", "content": "..." }
# { "type": "interrupt" }
# { "type": "memory_data", "memory": "...", "user": "..." }
# { "type": "skills_data", "skills": [...] }

# ---------------------------------------------------------------------------
# Message Types: Wasm → Host (stdout)
# ---------------------------------------------------------------------------

# { "type": "ready", "tools": [...], "toolsets": [...] }
# { "type": "api_request", "kwargs": { ...OpenAI API kwargs... } }
# { "type": "tool_call", "id": "...", "name": "...", "arguments": "..." }
#   → for REMOTE tools that need host execution
# { "type": "tool_response", "tool_call_id": "...", "name": "...", "content": "..." }
#   → for LOCAL tools executed inside Wasm
# { "type": "clarify_request", "question": "...", "choices": [...] }
# { "type": "assistant_message", "content": "..." }
# { "type": "status", "message": "..." }
# { "type": "error", "message": "..." }
# { "type": "done", "result": { ...conversation result... } }
# { "type": "request_memory" }
# { "type": "request_skills" }
# { "type": "compress_request", "messages": [...], "instruction": "..." }
#   → asks host to call LLM for summarization (context compression)
# { "type": "compress_response" }  ← host sends back the summary


def send(msg: dict):
    """Send a JSON message to the host (stdout)."""
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def recv() -> Optional[dict]:
    """Receive a JSON message from the host (stdin). Returns None on EOF."""
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())


def log(message: str, level: str = "info"):
    """Log a message to stderr (doesn't interfere with the JSON protocol)."""
    sys.stderr.write(f"[{level}] {message}\n")
    sys.stderr.flush()


def send_error(message: str):
    """Send an error message to the host."""
    send({"type": "error", "message": message})


def send_status(message: str):
    """Send a status update to the host."""
    send({"type": "status", "message": message})
