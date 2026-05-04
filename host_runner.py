#!/usr/bin/env python3
"""
Host runner for the hermes-pywasm agent.

This script manages the Wasm binary — feeding it user messages and
handling all I/O that the Wasm sandbox can't do:

  - LLM API calls (via openai or httpx)
  - User interaction (terminal I/O)
  - File persistence (memory files on disk)
  - Remote tool execution (subprocess-based tools)

The host and Wasm binary communicate via JSON lines over stdin/stdout.

Usage:
    # With compiled Wasm binary:
    python host_runner.py --wasm hermes_agent.wasm --model gpt-4

    # Development mode (runs wasm_agent.py directly as Python subprocess):
    python host_runner.py --dev --model gpt-4
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Host-side protocol helpers
# ---------------------------------------------------------------------------

def send_to_wasm(proc: subprocess.Popen, msg: dict):
    """Send a JSON message to the Wasm process via its stdin."""
    line = json.dumps(msg, ensure_ascii=False) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()


def recv_from_wasm(proc: subprocess.Popen) -> Optional[dict]:
    """Read a JSON message from the Wasm process via its stdout."""
    line = proc.stdout.readline()
    if not line:
        return None
    return json.loads(line.strip())


# ---------------------------------------------------------------------------
# LLM API caller
# ---------------------------------------------------------------------------

class LLMClient:
    """Makes OpenAI-compatible API calls.  Used by the host to fulfill
    api_request messages from the Wasm binary."""

    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-4"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        # Try openai library first, fall back to httpx/requests
        try:
            from openai import OpenAI
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
            self._mode = "openai"
        except ImportError:
            try:
                import httpx
                self._httpx = httpx
                self._mode = "httpx"
            except ImportError:
                import urllib.request
                self._mode = "urllib"

    def chat_completions_create(self, **kwargs) -> dict:
        """Call chat.completions.create and return the response as a dict."""
        if self._mode == "openai":
            response = self._client.chat.completions.create(**kwargs)
            return response.model_dump()

        # httpx/urllib fallback
        url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps(kwargs).encode()

        if self._mode == "httpx":
            resp = self._httpx.post(url, headers=headers, content=body, timeout=300)
            resp.raise_for_status()
            return resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Host-side tool dispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """Dispatches remote tool calls to the original hermes-agent tool implementations.

    Lazily imports tool modules from hermes-agent.  Falls back to an error
    message if the hermes-agent source isn't available.
    """

    def __init__(self, hermes_agent_dir: str = None):
        self._hermes_dir = hermes_agent_dir or str(
            Path(__file__).parent.parent / "hermes-agent"
        )
        self._registry = None
        self._available_tools: Set[str] = set()
        self._init_attempted = False

    def _ensure_init(self):
        """Lazy-init: import hermes-agent registry and trigger tool registration."""
        if self._init_attempted:
            return
        self._init_attempted = True

        hermes_path = Path(self._hermes_dir)
        if not hermes_path.exists():
            logger.warning("hermes-agent not found at %s — remote tools unavailable", self._hermes_dir)
            return

        # Add hermes-agent to path
        if self._hermes_dir not in sys.path:
            sys.path.insert(0, self._hermes_dir)

        try:
            # Add hermes-agent venv site-packages if available
            venv_site = hermes_path / "venv" / "lib"
            if venv_site.exists():
                for sp in venv_site.glob("python*/site-packages"):
                    if str(sp) not in sys.path:
                        sys.path.insert(0, str(sp))

            from tools.registry import registry

            # Import tool modules to trigger registration.
            # Each module may fail (missing deps) — that's fine, we just
            # won't have those tools available.
            _tool_modules = [
                "tools.terminal_tool",
                "tools.file_tools",
                "tools.skills_tool",
                "tools.skill_manager_tool",
                "tools.memory_tool",
                "tools.todo_tool",
                "tools.clarify_tool",
                "tools.code_execution_tool",
                "tools.delegate_tool",
                "tools.session_search_tool",
                "tools.cronjob_tools",
                "tools.web_tools",
                "tools.browser_tool",
                "tools.vision_tools",
                "tools.image_generation_tool",
                "tools.mixture_of_agents_tool",
                "tools.tts_tool",
            ]
            loaded = []
            failed = []
            for mod_name in _tool_modules:
                try:
                    __import__(mod_name)
                    loaded.append(mod_name.split(".")[-1])
                except Exception as e:
                    failed.append((mod_name.split(".")[-1], str(e)[:60]))
                    logger.debug("Could not import %s: %s", mod_name, e)

            if failed:
                logger.info("Tool modules that failed to load: %s",
                            ", ".join(f"{n} ({e})" for n, e in failed))

            self._registry = registry
            self._available_tools = set(registry.get_all_tool_names())
            logger.info("Loaded %d hermes-agent tools: %s",
                        len(self._available_tools),
                        ", ".join(sorted(self._available_tools)))

        except Exception as e:
            logger.warning("Failed to initialize hermes-agent tools: %s", e)

    @property
    def available_tools(self) -> List[str]:
        """Return list of tool names the host can execute."""
        self._ensure_init()
        return sorted(self._available_tools)

    @property
    def tool_schemas(self) -> List[dict]:
        """Return OpenAI-format tool schemas for all available tools.

        These are the *real* schemas from the hermes-agent registry,
        sent to the Wasm binary so it uses the correct parameter names.
        Bypasses check_fn so we get schemas even for tools whose runtime
        requirements aren't met (the host can still dispatch them).
        """
        self._ensure_init()
        if self._registry is None:
            return []
        result = []
        for name in sorted(self._available_tools):
            entry = self._registry._tools.get(name)
            if entry:
                result.append({"type": "function", "function": entry.schema})
        return result

    def dispatch(self, name: str, arguments: str) -> str:
        """Execute a tool by name with JSON arguments string.  Returns JSON result."""
        self._ensure_init()

        if self._registry is None:
            return json.dumps({
                "error": f"hermes-agent tools not available. "
                         f"Install hermes-agent at {self._hermes_dir}"
            })

        if name not in self._available_tools:
            return json.dumps({"error": f"Tool '{name}' not registered in hermes-agent."})

        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            args = {}

        try:
            result = self._registry.dispatch(name, args)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            logger.exception("Tool dispatch error for %s", name)
            return json.dumps({"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# Memory persistence
# ---------------------------------------------------------------------------

HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
MEMORY_DIR = HERMES_HOME / "memories"


def load_memory() -> Dict[str, str]:
    """Load memory files from disk."""
    result = {"memory": "", "user": ""}
    for target in ("memory", "user"):
        fname = "MEMORY.md" if target == "memory" else "USER.md"
        fpath = MEMORY_DIR / fname
        if fpath.exists():
            result[target] = fpath.read_text(encoding="utf-8").strip()
    return result


def save_memory(memory_text: str, user_text: str):
    """Save memory files to disk."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    (MEMORY_DIR / "MEMORY.md").write_text(memory_text, encoding="utf-8")
    (MEMORY_DIR / "USER.md").write_text(user_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main host loop
# ---------------------------------------------------------------------------

def run_host(
    wasm_path: str = None,
    dev_mode: bool = False,
    model: str = "gpt-4",
    api_key: str = None,
    base_url: str = None,
    enabled_toolsets: List[str] = None,
    user_message: str = None,
    state_dir: str = None,
):
    """Run the host process that manages the Wasm agent binary."""

    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: No API key. Set OPENAI_API_KEY or OPENROUTER_API_KEY.", file=sys.stderr)
        sys.exit(1)

    llm = LLMClient(api_key=api_key, base_url=base_url, model=model)

    # Initialize tool dispatcher for remote tool execution
    tool_dispatcher = ToolDispatcher()
    host_tools = tool_dispatcher.available_tools
    host_schemas = tool_dispatcher.tool_schemas
    if host_tools:
        print(f"Host tools available: {len(host_tools)} ({', '.join(host_tools[:10])}{'...' if len(host_tools) > 10 else ''})")
        print(f"Host schemas extracted: {len(host_schemas)}")

    # Launch the Wasm process
    if dev_mode:
        # Run wasm_agent.py directly as a Python subprocess
        agent_py = str(Path(__file__).parent / "wasm_agent.py")
        cmd = [sys.executable, agent_py]
    elif wasm_path:
        cmd = ["wasmtime"]
        if state_dir:
            cmd += [f"--dir={state_dir}::/state"]
        cmd += [wasm_path]
    else:
        print("Error: Specify --wasm <path> or --dev", file=sys.stderr)
        sys.exit(1)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # Step 1: Send init — tell Wasm which remote tools the host can execute,
        # and send the real schemas so tool parameters match exactly.
        config = {
            "model": model,
            "enabled_toolsets": enabled_toolsets or [],
            "context_length": 128000,
            "host_available_tools": host_tools,
            "host_tool_schemas": host_schemas,
        }
        send_to_wasm(proc, {"type": "init", "config": config})

        # Step 2: Wait for ready
        ready = recv_from_wasm(proc)
        if not ready or ready.get("type") != "ready":
            print(f"Error: Expected 'ready', got: {ready}", file=sys.stderr)
            sys.exit(1)

        print(f"Agent ready. Tools: {ready.get('tools', [])}")
        print(f"Local tools: {ready.get('local_tools', [])}")

        # Step 3: Conversation loop
        while True:
            # Get user input
            if user_message:
                msg_text = user_message
                user_message = None  # Only use CLI arg for first message
            else:
                try:
                    msg_text = input("\n› ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

            if not msg_text:
                continue
            if msg_text.lower() in ("quit", "exit", "/quit", "/exit"):
                break

            # Send user message
            send_to_wasm(proc, {"type": "user_message", "content": msg_text})

            # Process protocol messages until "done"
            while True:
                msg = recv_from_wasm(proc)
                if msg is None:
                    print("(agent process ended)", file=sys.stderr)
                    return

                msg_type = msg.get("type", "")

                if msg_type == "done":
                    result = msg.get("result", {})
                    if result.get("final_response"):
                        print(f"\n⚕ {result['final_response']}")
                    print(f"  ({result.get('api_calls', 0)} API calls)")
                    break

                elif msg_type == "api_request":
                    # Make the LLM API call
                    kwargs = msg.get("kwargs", {})
                    try:
                        response = llm.chat_completions_create(**kwargs)
                        send_to_wasm(proc, {"type": "api_response", "response": response})
                    except Exception as e:
                        send_to_wasm(proc, {"type": "error", "message": str(e)})

                elif msg_type == "request_memory":
                    mem = load_memory()
                    send_to_wasm(proc, {"type": "memory_data", **mem})

                elif msg_type == "request_skills":
                    # TODO: load skills from disk
                    send_to_wasm(proc, {"type": "skills_data", "skills": []})

                elif msg_type == "memory_update":
                    save_memory(msg.get("memory", ""), msg.get("user", ""))
                    print("  💾 Memory updated")

                elif msg_type == "tool_call":
                    # Remote tool execution via hermes-agent registry
                    name = msg.get("name", "")
                    arguments = msg.get("arguments", "{}")
                    print(f"  🔧 Remote tool: {name}")
                    try:
                        result_content = tool_dispatcher.dispatch(name, arguments)
                        # Truncate very long results for display
                        display = result_content[:200] + "..." if len(result_content) > 200 else result_content
                        print(f"  ┊ → {display}")
                    except Exception as e:
                        result_content = json.dumps({"error": f"Host dispatch failed: {e}"})
                        print(f"  ❌ {name}: {e}", file=sys.stderr)
                    send_to_wasm(proc, {
                        "type": "tool_result",
                        "tool_call_id": msg.get("id", ""),
                        "name": name,
                        "content": result_content,
                    })

                elif msg_type == "tool_response":
                    # Local tool result notification
                    name = msg.get("name", "")
                    print(f"  ┊ {name}: {msg.get('content', '')[:80]}")

                elif msg_type == "clarify_request":
                    # Ask user for clarification
                    q = msg.get("question", "")
                    choices = msg.get("choices")
                    print(f"\n❓ {q}")
                    if choices:
                        for i, c in enumerate(choices, 1):
                            print(f"  {i}. {c}")
                        print(f"  {len(choices)+1}. Other (type your answer)")
                    try:
                        answer = input("  › ").strip()
                        if choices and answer.isdigit():
                            idx = int(answer) - 1
                            if 0 <= idx < len(choices):
                                answer = choices[idx]
                    except (EOFError, KeyboardInterrupt):
                        answer = "(skipped)"
                    send_to_wasm(proc, {"type": "clarify_response", "content": answer})

                elif msg_type == "compress_request":
                    # Compression: call LLM to summarize
                    instruction = msg.get("instruction", "Summarize concisely.")
                    compress_msgs = msg.get("messages", [])
                    text = "\n".join(
                        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                        for m in compress_msgs
                    )
                    try:
                        resp = llm.chat_completions_create(
                            model=model,
                            messages=[
                                {"role": "system", "content": instruction},
                                {"role": "user", "content": text},
                            ],
                            max_tokens=500,
                        )
                        summary = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                        send_to_wasm(proc, {"type": "compress_response", "summary": summary})
                    except Exception as e:
                        send_to_wasm(proc, {"type": "compress_response", "summary": ""})

                elif msg_type == "assistant_message":
                    content = msg.get("content", "")
                    if content:
                        print(f"\n⚕ {content}")

                elif msg_type == "usage":
                    total = msg.get("total_tokens", 0)
                    if total:
                        print(f"  📊 {total:,} tokens")

                elif msg_type == "status":
                    print(f"  … {msg.get('message', '')}")

                elif msg_type == "error":
                    print(f"  ❌ {msg.get('message', '')}", file=sys.stderr)

                else:
                    print(f"  [unknown: {msg_type}]", file=sys.stderr)

    finally:
        # Shut down
        try:
            send_to_wasm(proc, {"type": "shutdown"})
        except Exception:
            pass
        proc.terminate()
        proc.wait(timeout=5)

        # Print any stderr from the Wasm process
        stderr = proc.stderr.read()
        if stderr.strip():
            print(f"\n[agent stderr]\n{stderr}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Host runner for hermes-pywasm agent")
    parser.add_argument("--wasm", type=str, help="Path to compiled .wasm binary")
    parser.add_argument("--dev", action="store_true", help="Run wasm_agent.py as Python subprocess")
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--toolsets", type=str, nargs="*", default=None)
    parser.add_argument("--message", "-m", type=str, default=None, help="Initial message (non-interactive)")
    parser.add_argument("--state-dir", type=str, default=None, help="WASI directory for state persistence")
    args = parser.parse_args()

    run_host(
        wasm_path=args.wasm,
        dev_mode=args.dev,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        enabled_toolsets=args.toolsets,
        user_message=args.message,
        state_dir=args.state_dir,
    )


if __name__ == "__main__":
    main()
