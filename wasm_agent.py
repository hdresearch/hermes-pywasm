"""
Hermes agent core loop — py2wasm-compatible.

This module contains the conversation loop logic extracted from run_agent.py,
rewritten to use only Python stdlib (no openai, httpx, pydantic, requests,
yaml, fire, dotenv).  All external I/O goes through wasm_protocol.py.

Compiles to WASI Wasm via py2wasm:
    py2wasm wasm_agent.py -o hermes_agent.wasm

Runs on wasmtime:
    echo '{"type":"init","config":{...}}' | wasmtime hermes_agent.wasm
"""

import copy
import hashlib
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import wasm_protocol as proto
from wasm_tools import (
    TOOL_REGISTRY,
    ALL_TOOL_NAMES,
    LOCAL_TOOL_NAMES,
    REMOTE_TOOL_NAMES,
    init_registry,
    get_tool_definitions,
    dispatch_local_tool,
    is_local_tool,
    is_interactive_tool,
    get_tool_locality,
)
from wasm_prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_system_prompt,
    build_memory_block,
    build_skills_block,
    build_todo_block,
)
from wasm_context import (
    estimate_tokens,
    estimate_messages_tokens,
    should_compress,
    get_messages_to_compress,
    apply_compression_result,
)


# =========================================================================
# Token estimation (pure Python, no tiktoken)
# =========================================================================

def estimate_tokens_rough(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)


# =========================================================================
# Tool call parsing
# =========================================================================

def parse_tool_calls(response: dict) -> List[dict]:
    """Extract tool calls from an OpenAI ChatCompletion response dict."""
    choices = response.get("choices", [])
    if not choices:
        return []
    message = choices[0].get("message", {})
    return message.get("tool_calls", []) or []


def get_assistant_content(response: dict) -> str:
    """Extract assistant text content from a response."""
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content") or ""


def get_finish_reason(response: dict) -> str:
    """Extract finish_reason from a response."""
    choices = response.get("choices", [])
    if not choices:
        return "stop"
    return choices[0].get("finish_reason", "stop") or "stop"


# =========================================================================
# Core agent loop
# =========================================================================

class WasmAgent:
    """Hermes agent conversation loop — pure Python, WASI-compatible."""

    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4")
        self.max_iterations = config.get("max_iterations", 25)
        self.enabled_toolsets = config.get("enabled_toolsets", [])
        self.disabled_toolsets = config.get("disabled_toolsets", [])
        self.quiet_mode = config.get("quiet_mode", False)
        self.system_prompt_override = config.get("system_prompt", None)
        self.temperature = config.get("temperature", None)
        self.top_p = config.get("top_p", None)
        self.max_tokens = config.get("max_tokens", None)
        self.context_length = config.get("context_length", 128000)
        self.compress_threshold = config.get("compress_threshold", 0.85)

        # Host-provided tool schemas and availability
        host_tool_schemas = config.get("host_tool_schemas", None)
        self.host_available_tools = config.get("host_available_tools", None)

        # Initialize the tool registry with host-provided schemas
        init_registry(host_tool_schemas=host_tool_schemas)

        # State
        self.session_id = config.get("session_id", str(uuid.uuid4()))
        self._interrupt_requested = False
        self._memory_text = ""
        self._user_text = ""
        self._skills_text = ""
        self._todo_items: List[dict] = []

        # Determine which tools to expose
        self._resolve_tools()

    def _resolve_tools(self):
        """Build the active tool set based on enabled/disabled toolsets
        and what the host reports it can execute."""
        all_tools = get_tool_definitions(
            enabled_toolsets=self.enabled_toolsets or None,
            disabled_toolsets=self.disabled_toolsets or None,
            host_available_tools=(
                set(self.host_available_tools) if self.host_available_tools else None
            ),
        )
        self.tools = all_tools
        self.tool_names = set()
        for tool_def in all_tools:
            self.tool_names.add(tool_def["function"]["name"])

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """Build the kwargs dict for the OpenAI chat.completions.create call."""
        kwargs = {
            "model": self.model,
            "messages": api_messages,
        }

        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        return kwargs

    def _request_api_call(self, api_kwargs: dict) -> dict:
        """Send API request to host, wait for response."""
        proto.send({"type": "api_request", "kwargs": api_kwargs})

        while True:
            msg = proto.recv()
            if msg is None:
                raise EOFError("Host disconnected")

            if msg["type"] == "api_response":
                return msg["response"]
            elif msg["type"] == "interrupt":
                self._interrupt_requested = True
                raise InterruptedError("Agent interrupted during API call")
            elif msg["type"] == "error":
                raise RuntimeError(f"Host error: {msg.get('message', 'unknown')}")
            else:
                proto.log(f"Unexpected message during API wait: {msg['type']}", "warn")

    def _request_remote_tool(self, tool_call: dict) -> str:
        """Ask host to execute a remote tool, wait for result."""
        fn = tool_call.get("function", {})
        proto.send({
            "type": "tool_call",
            "id": tool_call.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        })

        while True:
            msg = proto.recv()
            if msg is None:
                raise EOFError("Host disconnected")

            if msg["type"] == "tool_result":
                return msg.get("content", "")
            elif msg["type"] == "interrupt":
                self._interrupt_requested = True
                return json.dumps({"error": "Interrupted"})
            else:
                proto.log(f"Unexpected message during tool wait: {msg['type']}", "warn")

    def _request_clarify(self, question: str, choices: Optional[list] = None) -> str:
        """Ask host to present a clarification question to the user."""
        proto.send({
            "type": "clarify_request",
            "question": question,
            "choices": choices,
        })

        while True:
            msg = proto.recv()
            if msg is None:
                raise EOFError("Host disconnected")

            if msg["type"] == "clarify_response":
                return msg.get("content", "")
            elif msg["type"] == "interrupt":
                self._interrupt_requested = True
                return "(interrupted)"
            else:
                proto.log(f"Unexpected message during clarify wait: {msg['type']}", "warn")

    def _request_compression(self, messages: list, instruction: str) -> str:
        """Ask host to call LLM for context compression."""
        proto.send({
            "type": "compress_request",
            "messages": messages,
            "instruction": instruction,
        })

        while True:
            msg = proto.recv()
            if msg is None:
                raise EOFError("Host disconnected")

            if msg["type"] == "compress_response":
                return msg.get("summary", "")
            elif msg["type"] == "interrupt":
                self._interrupt_requested = True
                return ""
            else:
                proto.log(f"Unexpected message during compress wait: {msg['type']}", "warn")

    def _load_memory(self):
        """Request memory data from host."""
        proto.send({"type": "request_memory"})
        msg = proto.recv()
        if msg and msg["type"] == "memory_data":
            self._memory_text = msg.get("memory", "")
            self._user_text = msg.get("user", "")

    def _load_skills(self):
        """Request skills data from host."""
        proto.send({"type": "request_skills"})
        msg = proto.recv()
        if msg and msg["type"] == "skills_data":
            skills = msg.get("skills", [])
            self._skills_text = build_skills_block(skills) if skills else ""

    def _build_full_system_prompt(self, override: str = None) -> str:
        """Build the complete system prompt with memory, skills, etc."""
        base = override or self.system_prompt_override or DEFAULT_SYSTEM_PROMPT

        parts = [base]

        if self._memory_text:
            parts.append(build_memory_block(self._memory_text))
        if self._user_text:
            parts.append(f"\n## User Notes\n{self._user_text}")
        if self._skills_text:
            parts.append(self._skills_text)
        if self._todo_items:
            parts.append(build_todo_block(self._todo_items))

        return "\n\n".join(parts)

    def _execute_tool_calls(
        self, tool_calls: list, messages: list
    ) -> list:
        """Process tool calls: dispatch local ones, delegate remote ones."""
        results = []

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            fn_args_str = fn.get("arguments", "{}")
            tc_id = tc.get("id", str(uuid.uuid4()))

            # Parse arguments
            try:
                fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
            except json.JSONDecodeError:
                fn_args = {}

            proto.send_status(f"Executing tool: {fn_name}")

            if fn_name == "clarify":
                # Special case: clarify goes through host for user interaction
                user_resp = self._request_clarify(
                    fn_args.get("question", ""),
                    fn_args.get("choices"),
                )
                result_content = json.dumps({
                    "question": fn_args.get("question", ""),
                    "choices_offered": fn_args.get("choices"),
                    "user_response": user_resp,
                }, ensure_ascii=False)

            elif is_local_tool(fn_name):
                # Execute locally inside Wasm
                result_content = dispatch_local_tool(
                    fn_name, fn_args,
                    todo_items=self._todo_items,
                    memory_text=self._memory_text,
                    user_text=self._user_text,
                )

                # If todo tool was used, update our state
                if fn_name == "todo":
                    try:
                        result_data = json.loads(result_content)
                        if "todos" in result_data:
                            self._todo_items = result_data["todos"]
                    except (json.JSONDecodeError, KeyError):
                        pass

                # If memory tool mutated, tell host to persist
                if fn_name == "memory":
                    try:
                        result_data = json.loads(result_content)
                        if result_data.get("success"):
                            proto.send({
                                "type": "memory_update",
                                "memory": self._memory_text,
                                "user": self._user_text,
                            })
                    except (json.JSONDecodeError, KeyError):
                        pass

            else:
                # Remote tool: delegate to host
                result_content = self._request_remote_tool(tc)

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc_id,
                "name": fn_name,
                "content": result_content,
            }
            results.append(tool_msg)
            messages.append(tool_msg)

            # Notify host of local tool results
            proto.send({
                "type": "tool_response",
                "tool_call_id": tc_id,
                "name": fn_name,
                "content": result_content,
            })

        return results

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the full conversation loop.

        Returns a result dict matching the original run_agent.py format.
        """
        # Load memory and skills from host
        self._load_memory()
        self._load_skills()

        # Initialize messages
        messages = list(conversation_history) if conversation_history else []
        messages.append({"role": "user", "content": user_message})

        # Build system prompt
        system_prompt = self._build_full_system_prompt(system_message)

        # Main loop
        api_call_count = 0
        final_response = None
        interrupted = False

        self._interrupt_requested = False

        threshold_tokens = int(self.context_length * self.compress_threshold)

        while api_call_count < self.max_iterations:
            if self._interrupt_requested:
                interrupted = True
                break

            api_call_count += 1
            proto.send_status(
                f"API call #{api_call_count}/{self.max_iterations} "
                f"({len(messages)} messages)"
            )

            # Build API messages with system prompt
            api_messages = [{"role": "system", "content": system_prompt}] + messages

            # Check if we need context compression
            total_tokens = estimate_messages_tokens(api_messages)
            if total_tokens > threshold_tokens and len(messages) > 4:
                proto.send_status(
                    f"Context compression needed: ~{total_tokens:,} tokens "
                    f"> {threshold_tokens:,} threshold"
                )
                compress_msgs, keep_before, keep_after = get_messages_to_compress(
                    messages, protect_first=2, protect_last=2
                )
                if compress_msgs:
                    summary = self._request_compression(
                        compress_msgs,
                        "Summarize the following conversation excerpt concisely, "
                        "preserving key facts, decisions, and tool results.",
                    )
                    if summary:
                        messages = apply_compression_result(
                            messages, keep_before, keep_after, summary
                        )
                        api_messages = [{"role": "system", "content": system_prompt}] + messages
                        proto.send_status(
                            f"Compressed to {len(messages)} messages"
                        )

            # Build API kwargs and request
            api_kwargs = self._build_api_kwargs(api_messages)

            try:
                response = self._request_api_call(api_kwargs)
            except InterruptedError:
                interrupted = True
                final_response = "Operation interrupted."
                break
            except Exception as e:
                proto.send_error(f"API call failed: {e}")
                final_response = f"API call failed: {e}"
                break

            # Parse response
            finish_reason = get_finish_reason(response)
            tool_calls = parse_tool_calls(response)
            content = get_assistant_content(response)

            # Track token usage
            usage = response.get("usage", {})
            if usage:
                proto.send({
                    "type": "usage",
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                })

            if tool_calls:
                # Build assistant message with tool calls
                assistant_msg = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_msg)

                if content:
                    proto.send({"type": "assistant_message", "content": content})

                # Execute tool calls
                self._execute_tool_calls(tool_calls, messages)

                # Continue the loop for the next API call
                continue

            else:
                # No tool calls — this is the final response
                final_response = content or ""
                messages.append({"role": "assistant", "content": final_response})
                proto.send({"type": "assistant_message", "content": final_response})
                break

        # Handle max iterations
        if api_call_count >= self.max_iterations and final_response is None:
            final_response = f"Reached maximum iterations ({self.max_iterations})."

        result = {
            "final_response": final_response,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": not interrupted and final_response is not None,
            "interrupted": interrupted,
            "session_id": self.session_id,
        }

        return result


# =========================================================================
# Main entry point
# =========================================================================

def main():
    """Main loop: init, then process user messages until EOF."""
    proto.log("Hermes agent (WASI) starting...")

    # Wait for init message
    init_msg = proto.recv()
    if init_msg is None or init_msg.get("type") != "init":
        proto.send_error("Expected init message")
        sys.exit(1)

    config = init_msg.get("config", {})
    agent = WasmAgent(config)

    # Tell host we're ready — report which tools are local vs remote
    tool_names = [t["function"]["name"] for t in agent.tools]
    local = [n for n in tool_names if is_local_tool(n)]
    remote = [n for n in tool_names if not is_local_tool(n) and not is_interactive_tool(n)]
    interactive = [n for n in tool_names if is_interactive_tool(n)]
    proto.send({
        "type": "ready",
        "tools": tool_names,
        "local_tools": local,
        "remote_tools": remote,
        "interactive_tools": interactive,
        "session_id": agent.session_id,
    })

    # Process messages
    while True:
        msg = proto.recv()
        if msg is None:
            break  # EOF

        if msg["type"] == "user_message":
            result = agent.run_conversation(
                user_message=msg.get("content", ""),
                system_message=msg.get("system_message"),
                conversation_history=msg.get("conversation_history"),
            )
            proto.send({"type": "done", "result": result})

        elif msg["type"] == "interrupt":
            agent._interrupt_requested = True

        elif msg["type"] == "shutdown":
            break

        else:
            proto.log(f"Unknown top-level message type: {msg['type']}", "warn")

    proto.log("Hermes agent (WASI) shutting down.")


if __name__ == "__main__":
    main()
