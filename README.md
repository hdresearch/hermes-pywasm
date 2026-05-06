![Bringing Hermes to WebAssembly](hermes-wasm-cover.png)

# hermes-pywasm

Run [hermes-agent](../hermes-agent) as a compiled WebAssembly binary via [py2wasm](https://github.com/nicolo-ribaudo/nicolo-ribaudo.github.io) (Nuitka → C → WASI Wasm).

Unlike [hermes-pyodide](../hermes-pyodide) which monkey-patches the original code to run in a browser, this is a **ground-up rewrite** of the core agent loop using only Python stdlib.  The result is a 26MB `.wasm` binary that runs on wasmtime/wasmer — no Python interpreter needed at runtime.

## Architecture: Host ↔ Wasm Split

py2wasm compiles to WASI, which has **no networking, no threads, no C extensions, no asyncio**.  The solution is a split architecture:

```
┌────────────────────────────────┐      JSON lines        ┌─────────────────────────────────┐
│          HOST PROCESS          │    ◄──── stdin ────►    │        WASM BINARY              │
│                                │    ◄──── stdout ───►    │                                 │
│  • LLM API calls (httpx)       │                         │  • Conversation loop             │
│  • User I/O (terminal/web)     │                         │  • System prompt building        │
│  • File persistence            │                         │  • Tool call parsing             │
│  • Remote tool execution       │                         │  • Local tool dispatch           │
│  • Context compression (LLM)   │                         │    (memory, todo)               │
│                                │                         │  • Token estimation              │
│  host_runner.py                │                         │  • Message history management    │
│  (Python, Node, Go, Rust...)   │                         │  • Context compression decisions │
│                                │                         │                                 │
│                                │                         │  wasm_agent.py → hermes_agent.wasm│
│                                │                         │  (26MB, pure Python stdlib)      │
└────────────────────────────────┘                         └─────────────────────────────────┘
```

The Wasm binary handles all **computation** (the agent "brain").  The host handles all **I/O** (the agent "body").

## Protocol

Communication is newline-delimited JSON on stdin/stdout.  Each message has a `type` field.

### Host → Wasm (stdin)
| Type | Purpose |
|------|---------|
| `init` | Configuration (model, toolsets, context_length) |
| `user_message` | User's text input |
| `api_response` | LLM API response (ChatCompletion JSON) |
| `memory_data` | Memory/user text from disk |
| `skills_data` | Available skills list |
| `tool_result` | Result of a remote tool executed by host |
| `clarify_response` | User's answer to a clarification question |
| `compress_response` | Summary text from LLM compression call |
| `interrupt` | Cancel current operation |
| `shutdown` | Graceful shutdown |

### Wasm → Host (stdout)
| Type | Purpose |
|------|---------|
| `ready` | Agent initialized, lists available tools |
| `api_request` | Request to make LLM API call (full kwargs) |
| `tool_call` | Request to execute a remote tool |
| `tool_response` | Result of locally-executed tool |
| `clarify_request` | Question for the user |
| `compress_request` | Request to call LLM for context summarization |
| `assistant_message` | Text output from the assistant |
| `memory_update` | Mutated memory to persist |
| `usage` | Token usage stats |
| `status` | Progress updates |
| `error` | Error messages |
| `done` | Conversation complete, final result |

## What's inside

| File | Purpose | Lines |
|------|---------|-------|
| `wasm_agent.py` | Core conversation loop | ~380 |
| `wasm_protocol.py` | JSON-lines stdin/stdout protocol | ~90 |
| `wasm_tools.py` | Tool registry + local tool implementations (memory, todo) | ~340 |
| `wasm_prompts.py` | System prompt construction | ~100 |
| `wasm_context.py` | Token estimation, compression decisions | ~100 |
| `host_runner.py` | Host-side process that manages the Wasm binary | ~290 |
| `build.sh` | Build + test script | ~60 |
| **Total** | | **~1,360** |

## Build

```bash
# Install py2wasm (requires Python 3.11)
python3.11 -m pip install py2wasm

# Compile to Wasm
./build.sh

# Test (Python protocol tests + Wasm smoke test)
./build.sh test
```

The output is `hermes_agent.wasm` (26MB).

## Run

```bash
# Development mode (runs Python, no compilation needed)
python3 host_runner.py --dev --model gpt-4

# Production mode (runs compiled Wasm binary)
python3 host_runner.py --wasm hermes_agent.wasm --model gpt-4

# Single message (non-interactive)
python3 host_runner.py --dev --model gpt-4 -m "What is 2+2?"

# With wasmtime directly (for embedding in non-Python hosts)
printf '{"type":"init","config":{"model":"gpt-4"}}\n...' | wasmtime hermes_agent.wasm
```

## Local tools (run inside Wasm)

These tools execute entirely within the Wasm sandbox:

| Tool | What it does |
|------|-------------|
| `memory` | Bounded curated memory (add/replace/remove/read) with char limits |
| `todo` | In-session task list management |

## Remote tools (delegated to host)

These tools require host execution (networking, subprocess, etc.):

| Tool | Why remote |
|------|-----------|
| `clarify` | Needs user I/O |
| `terminal` | Needs subprocess |
| `browser_*` | Needs Browserbase |
| `web_search/extract/crawl` | Needs HTTP + Firecrawl |
| `vision_analyze` | Needs HTTP |
| `image_generate` | Needs HTTP |
| All others | Various I/O requirements |

## Limitations vs. original hermes-agent

| Feature | Original | hermes-pywasm |
|---------|----------|---------------|
| Full tool suite (45 tools) | ✅ | 2 local + host-delegated |
| Session DB (SQLite + FTS5) | ✅ | ❌ (no sqlite3 in WASI) |
| Prompt caching (Anthropic) | ✅ | ❌ (in API kwargs but not Wasm-side cache control) |
| Context compression | ✅ (LLM call) | ✅ (via host delegation) |
| Streaming responses | ✅ | ❌ (full response only) |
| Token counting (tiktoken) | ✅ | ~rough estimate (chars/4) |
| pydantic validation | ✅ | ❌ (pure dict validation) |
| YAML config | ✅ | ❌ (JSON config only) |
| .env loading | ✅ | ❌ (host provides config) |
| Gateway (Telegram/Discord) | ✅ | Host wraps protocol |
