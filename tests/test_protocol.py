#!/usr/bin/env python3
"""
Test the wasm_agent protocol end-to-end.

Runs wasm_agent.py as a subprocess and communicates via JSON lines,
simulating what the host_runner.py does but with mock API responses.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).parent.parent


def send(proc, msg):
    line = json.dumps(msg, ensure_ascii=False) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()


def recv(proc, timeout=30):
    """Read one JSON line from the agent.  Blocks up to timeout seconds.

    Uses a background thread because select() doesn't work reliably
    with Python TextIOWrapper buffering.
    """
    import threading
    result = [None]
    error = [None]

    def _read():
        try:
            line = proc.stdout.readline()
            if line:
                result[0] = json.loads(line.strip())
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError("No response from agent within timeout")
    if error[0]:
        raise error[0]
    return result[0]


def recv_until(proc, target_type, timeout=30):
    """Read messages until we get one of the target type."""
    messages = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = recv(proc, timeout=max(0.1, deadline - time.time()))
        if msg is None:
            break
        messages.append(msg)
        if msg.get("type") == target_type:
            return msg, messages
    raise TimeoutError(f"Never received '{target_type}' message. Got: {[m.get('type') for m in messages]}")


def make_mock_response(content: str, tool_calls=None):
    """Build a mock ChatCompletion response dict."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def test_basic_conversation():
    """Test: init → user_message → api_request → api_response → done"""
    print("1. Testing basic conversation flow...")

    proc = subprocess.Popen(
        [sys.executable, "-u", str(AGENT_DIR / "wasm_agent.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(AGENT_DIR),
    )

    try:
        # Init
        send(proc, {"type": "init", "config": {"model": "test-model", "max_iterations": 5}})

        # Wait for ready
        ready, _ = recv_until(proc, "ready")
        assert ready["type"] == "ready"
        assert "memory" in ready["tools"]
        assert "todo" in ready["tools"]
        print(f"   ✓ Ready received. Tools: {ready['tools']}")

        # Send user message
        send(proc, {"type": "user_message", "content": "Hello, how are you?"})

        # Should get request_memory first
        mem_req, _ = recv_until(proc, "request_memory")
        send(proc, {"type": "memory_data", "memory": "", "user": ""})

        # Then request_skills
        skills_req, _ = recv_until(proc, "request_skills")
        send(proc, {"type": "skills_data", "skills": []})

        # Then api_request
        api_req, _ = recv_until(proc, "api_request")
        assert "kwargs" in api_req
        kwargs = api_req["kwargs"]
        assert kwargs["model"] == "test-model"
        assert any("Hello" in str(m.get("content", "")) for m in kwargs["messages"])
        print(f"   ✓ API request received. {len(kwargs['messages'])} messages, {len(kwargs.get('tools', []))} tools")

        # Send mock response
        send(proc, {"type": "api_response", "response": make_mock_response("I'm doing great! How can I help you?")})

        # Should get done
        done, intermediates = recv_until(proc, "done")
        result = done["result"]
        assert result["completed"] is True
        assert result["api_calls"] == 1
        assert "great" in result["final_response"].lower()
        print(f"   ✓ Done. Response: '{result['final_response'][:60]}'")
        print(f"   ✓ API calls: {result['api_calls']}, completed: {result['completed']}")

    finally:
        proc.terminate()
        proc.wait(timeout=5)

    print("   ✅ Basic conversation: PASSED\n")


def test_tool_call_flow():
    """Test: tool call → local dispatch → follow-up API call → done"""
    print("2. Testing tool call flow (memory + todo)...")

    proc = subprocess.Popen(
        [sys.executable, "-u", str(AGENT_DIR / "wasm_agent.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(AGENT_DIR),
    )

    try:
        # Init
        send(proc, {"type": "init", "config": {"model": "test-model", "max_iterations": 10}})
        ready, _ = recv_until(proc, "ready")

        # Send user message
        send(proc, {"type": "user_message", "content": "Remember that my cat is named Luna"})

        # Handle memory/skills requests
        mem_req, _ = recv_until(proc, "request_memory")
        send(proc, {"type": "memory_data", "memory": "Previous fact: user likes Python", "user": ""})
        skills_req, _ = recv_until(proc, "request_skills")
        send(proc, {"type": "skills_data", "skills": []})

        # API request #1
        api_req, _ = recv_until(proc, "api_request")

        # Respond with a memory tool call
        tool_call_response = make_mock_response(
            "I'll save that to my memories!",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "memory",
                    "arguments": json.dumps({
                        "action": "add",
                        "target": "user",
                        "content": "Cat's name is Luna"
                    }),
                },
            }]
        )
        send(proc, {"type": "api_response", "response": tool_call_response})

        # Agent should execute memory tool locally and then make another API call
        # We need to collect messages until we see another api_request
        messages = []
        deadline = time.time() + 10
        api_req_2 = None
        while time.time() < deadline:
            msg = recv(proc, timeout=5)
            if msg is None:
                break
            messages.append(msg)
            if msg["type"] == "api_request":
                api_req_2 = msg
                break

        assert api_req_2 is not None, f"Expected second api_request, got: {[m['type'] for m in messages]}"

        # Check that tool_response was sent
        tool_responses = [m for m in messages if m["type"] == "tool_response"]
        assert len(tool_responses) >= 1, "Should have tool_response"
        assert tool_responses[0]["name"] == "memory"
        tool_result = json.loads(tool_responses[0]["content"])
        assert tool_result["success"] is True
        print(f"   ✓ Memory tool executed locally: {tool_result.get('action', 'add')}")

        # Send final response
        send(proc, {"type": "api_response", "response": make_mock_response(
            "Done! I've saved that your cat's name is Luna."
        )})

        done, _ = recv_until(proc, "done")
        result = done["result"]
        assert result["completed"] is True
        assert result["api_calls"] == 2
        print(f"   ✓ Conversation completed in {result['api_calls']} API calls")

    finally:
        proc.terminate()
        proc.wait(timeout=5)

    print("   ✅ Tool call flow: PASSED\n")


def test_todo_tool():
    """Test the todo tool (local, pure Python)."""
    print("3. Testing todo tool...")

    proc = subprocess.Popen(
        [sys.executable, "-u", str(AGENT_DIR / "wasm_agent.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(AGENT_DIR),
    )

    try:
        send(proc, {"type": "init", "config": {"model": "test-model"}})
        ready, _ = recv_until(proc, "ready")

        send(proc, {"type": "user_message", "content": "Create a todo list for my project"})

        # Handle prereqs
        mem_req, _ = recv_until(proc, "request_memory")
        send(proc, {"type": "memory_data", "memory": "", "user": ""})
        skills_req, _ = recv_until(proc, "request_skills")
        send(proc, {"type": "skills_data", "skills": []})

        api_req, _ = recv_until(proc, "api_request")

        # Respond with todo tool call
        send(proc, {"type": "api_response", "response": make_mock_response(
            "Let me create a task list.",
            tool_calls=[{
                "id": "call_todo",
                "type": "function",
                "function": {
                    "name": "todo",
                    "arguments": json.dumps({
                        "todos": [
                            {"id": "1", "content": "Set up project structure", "status": "in_progress"},
                            {"id": "2", "content": "Write tests", "status": "pending"},
                            {"id": "3", "content": "Deploy to production", "status": "pending"},
                        ]
                    }),
                },
            }]
        )})

        # Collect until next api_request
        messages = []
        deadline = time.time() + 10
        while time.time() < deadline:
            msg = recv(proc, timeout=5)
            if msg is None:
                break
            messages.append(msg)
            if msg["type"] == "api_request":
                break

        tool_responses = [m for m in messages if m["type"] == "tool_response"]
        assert len(tool_responses) >= 1
        todo_result = json.loads(tool_responses[0]["content"])
        assert todo_result["count"] == 3
        assert todo_result["todos"][0]["content"] == "Set up project structure"
        print(f"   ✓ Todo tool: {todo_result['count']} items created")

        # Final response
        send(proc, {"type": "api_response", "response": make_mock_response(
            "I've created your todo list with 3 items."
        )})

        done, _ = recv_until(proc, "done")
        assert done["result"]["completed"] is True
        print(f"   ✓ Completed: {done['result']['final_response'][:60]}")

    finally:
        proc.terminate()
        proc.wait(timeout=5)

    print("   ✅ Todo tool: PASSED\n")


def test_multi_turn():
    """Test multiple user messages in one session."""
    print("4. Testing multi-turn conversation...")

    proc = subprocess.Popen(
        [sys.executable, "-u", str(AGENT_DIR / "wasm_agent.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(AGENT_DIR),
    )

    try:
        send(proc, {"type": "init", "config": {"model": "test-model", "max_iterations": 5}})
        ready, _ = recv_until(proc, "ready")

        for turn, (user_text, bot_text) in enumerate([
            ("What is 2+2?", "2+2 equals 4."),
            ("And 3+3?", "3+3 equals 6."),
            ("Thanks!", "You're welcome!"),
        ], 1):
            send(proc, {"type": "user_message", "content": user_text})

            # Handle memory/skills
            mem_req, _ = recv_until(proc, "request_memory")
            send(proc, {"type": "memory_data", "memory": "", "user": ""})
            skills_req, _ = recv_until(proc, "request_skills")
            send(proc, {"type": "skills_data", "skills": []})

            api_req, _ = recv_until(proc, "api_request")
            send(proc, {"type": "api_response", "response": make_mock_response(bot_text)})

            done, _ = recv_until(proc, "done")
            assert done["result"]["completed"] is True
            print(f"   ✓ Turn {turn}: '{user_text}' → '{bot_text}'")

    finally:
        proc.terminate()
        proc.wait(timeout=5)

    print("   ✅ Multi-turn: PASSED\n")


def test_host_schemas():
    """Test that host-provided tool schemas are used correctly."""
    print("5. Testing host-provided tool schemas...")

    proc = subprocess.Popen(
        [sys.executable, "-u", str(AGENT_DIR / "wasm_agent.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(AGENT_DIR),
    )

    # Fake schemas that the host would extract from hermes-agent registry
    fake_schemas = [
        {"type": "function", "function": {
            "name": "terminal",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 180},
                },
                "required": ["command"],
            },
        }},
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "default": 1},
                    "limit": {"type": "integer", "default": 500},
                },
                "required": ["path"],
            },
        }},
    ]

    try:
        send(proc, {"type": "init", "config": {
            "model": "test-model",
            "host_available_tools": ["terminal", "read_file"],
            "host_tool_schemas": fake_schemas,
        }})
        ready, _ = recv_until(proc, "ready")
        tools = ready["tools"]
        local = ready["local_tools"]
        remote = ready["remote_tools"]

        # Should have: memory, todo (local) + clarify (interactive) + terminal, read_file (remote)
        assert "memory" in tools, f"memory missing: {tools}"
        assert "todo" in tools, f"todo missing: {tools}"
        assert "clarify" in tools, f"clarify missing: {tools}"
        assert "terminal" in tools, f"terminal missing: {tools}"
        assert "read_file" in tools, f"read_file missing: {tools}"
        assert len(tools) == 5, f"Expected 5 tools, got {len(tools)}: {tools}"
        assert "terminal" in remote, f"terminal should be remote: {remote}"
        assert "read_file" in remote, f"read_file should be remote: {remote}"
        print(f"   ✓ Got {len(tools)} tools: {len(local)} local + {len(remote)} remote + {len(ready['interactive_tools'])} interactive")

        # Send a message that triggers a read_file tool call
        send(proc, {"type": "user_message", "content": "Read a file"})

        mem_req, _ = recv_until(proc, "request_memory")
        send(proc, {"type": "memory_data", "memory": "", "user": ""})
        skills_req, _ = recv_until(proc, "request_skills")
        send(proc, {"type": "skills_data", "skills": []})

        api_req, _ = recv_until(proc, "api_request")
        kwargs = api_req["kwargs"]

        # Verify the API request uses the HOST's schema (path, not file_path)
        tool_defs = kwargs.get("tools", [])
        rf_def = None
        for td in tool_defs:
            if td["function"]["name"] == "read_file":
                rf_def = td
                break
        assert rf_def is not None, "read_file not in API tools"
        params = list(rf_def["function"]["parameters"]["properties"].keys())
        assert "path" in params, f"Expected 'path' param, got {params}"
        assert "file_path" not in params, f"'file_path' should not be in params: {params}"
        print(f"   ✓ read_file uses host schema (params: {params})")

        # Complete the conversation
        send(proc, {"type": "api_response", "response": make_mock_response("Here's the file content.")})
        done, _ = recv_until(proc, "done")
        assert done["result"]["completed"] is True
        print(f"   ✓ Conversation completed")

    finally:
        proc.terminate()
        proc.wait(timeout=5)

    print("   ✅ Host schemas: PASSED\n")


def test_remote_tool_dispatch():
    """Test remote tool call flow with host-provided schemas."""
    print("6. Testing remote tool dispatch...")

    proc = subprocess.Popen(
        [sys.executable, "-u", str(AGENT_DIR / "wasm_agent.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=str(AGENT_DIR),
    )

    fake_schemas = [
        {"type": "function", "function": {
            "name": "terminal",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }},
    ]

    try:
        send(proc, {"type": "init", "config": {
            "model": "test-model",
            "host_available_tools": ["terminal"],
            "host_tool_schemas": fake_schemas,
        }})
        ready, _ = recv_until(proc, "ready")

        send(proc, {"type": "user_message", "content": "Run ls"})

        mem_req, _ = recv_until(proc, "request_memory")
        send(proc, {"type": "memory_data", "memory": "", "user": ""})
        skills_req, _ = recv_until(proc, "request_skills")
        send(proc, {"type": "skills_data", "skills": []})

        api_req, _ = recv_until(proc, "api_request")

        # Simulate LLM calling terminal (remote tool)
        tool_call_response = make_mock_response(
            "Let me run that command.",
            tool_calls=[{
                "id": "call_t1",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": "ls -la"})
                }
            }]
        )
        # Override finish_reason
        tool_call_response["choices"][0]["finish_reason"] = "tool_calls"
        send(proc, {"type": "api_response", "response": tool_call_response})

        # Agent should emit a tool_call for terminal (remote)
        tool_call, intermediates = recv_until(proc, "tool_call")
        assert tool_call["name"] == "terminal", f"Expected terminal, got {tool_call['name']}"
        args = json.loads(tool_call["arguments"])
        assert args.get("command") == "ls -la", f"Expected 'ls -la', got {args}"
        print(f"   ✓ Remote tool_call received: terminal({args})")

        # Send tool result back
        send(proc, {
            "type": "tool_result",
            "tool_call_id": "call_t1",
            "name": "terminal",
            "content": json.dumps({"output": "file1.py\nfile2.py", "exit_code": 0}),
        })

        # Agent makes 2nd API call
        api_req2, _ = recv_until(proc, "api_request")
        assert len(api_req2["kwargs"]["messages"]) > 2, "Should have tool result in messages"
        print(f"   ✓ 2nd API request with {len(api_req2['kwargs']['messages'])} messages")

        send(proc, {"type": "api_response", "response": make_mock_response(
            "The directory contains file1.py and file2.py."
        )})

        done, _ = recv_until(proc, "done")
        assert done["result"]["completed"] is True
        assert done["result"]["api_calls"] == 2
        print(f"   ✓ Done: {done['result']['api_calls']} API calls")

    finally:
        proc.terminate()
        proc.wait(timeout=5)

    print("   ✅ Remote tool dispatch: PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("hermes-pywasm protocol tests")
    print("=" * 60)
    print()

    test_basic_conversation()
    test_tool_call_flow()
    test_todo_tool()
    test_multi_turn()
    test_host_schemas()
    test_remote_tool_dispatch()

    print("=" * 60)
    print("✅ All protocol tests passed!")
    print("=" * 60)
