# pywasm Tool Calling — What Changed

## Before

- **2 local tools** (memory, todo) running inside Wasm sandbox
- **1 interactive tool** (clarify) delegated to host for user interaction
- **0 remote tools** — host stubbed all with `"not implemented"` errors
- LLM only knew about 3 tools, couldn't plan beyond memory/todo/clarify

## After

- **2 local tools** (memory, todo) — unchanged, run inside Wasm sandbox
- **1 interactive tool** (clarify) — unchanged, host handles user interaction
- **28 remote tools** — full schemas baked into Wasm, host dispatches to real hermes-agent implementations
- **31 total tools** exposed to the LLM — it can plan using terminal, file ops, web search, browser, vision, code execution, skills, delegation, etc.

## What Was Implemented

### 1. Full Tool Schema Registry (`wasm_tools.py`)

All 31 tool schemas are embedded in the Wasm binary at compile time:
- `terminal`, `process` — shell execution
- `read_file`, `write_file`, `patch`, `search_files` — file operations
- `web_search`, `web_extract` — web research
- `browser_navigate/snapshot/click/type/scroll/back/press/close/get_images/vision` — browser automation
- `skills_list`, `skill_view`, `skill_manage` — skill management
- `vision_analyze` — image analysis
- `image_generate` — DALL-E image generation
- `execute_code` — Python script execution
- `delegate_task` — sub-agent spawning
- `mixture_of_agents` — multi-model reasoning
- `session_search` — past conversation search
- `text_to_speech` — TTS

### 2. Host Tool Dispatcher (`host_runner.py`)

New `ToolDispatcher` class that:
- Lazily imports hermes-agent tool modules (terminal, file, web, browser, etc.)
- Uses the hermes-agent `ToolRegistry` for dispatch (same registry the native agent uses)
- Adds hermes-agent venv site-packages to sys.path automatically
- Reports available tools to Wasm binary at init time
- Handles failures gracefully (missing deps → tool unavailable, not crash)

### 3. Host-Available Tool Filtering

The init protocol now supports `host_available_tools`:
```json
{"type": "init", "config": {
    "model": "gpt-4",
    "host_available_tools": ["terminal", "read_file", "web_search"]
}}
```

The Wasm agent filters its tool definitions to only expose tools the host
can actually execute, plus all local tools (always available).

### 4. Ready Message Enhancement

The ready message now reports tool locality:
```json
{"type": "ready",
 "tools": ["memory", "todo", "clarify", "terminal", "web_search", ...],
 "local_tools": ["memory", "todo"],
 "remote_tools": ["terminal", "web_search", ...],
 "interactive_tools": ["clarify"]}
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Wasm Sandbox (hermes_agent.wasm)                   │
│                                                     │
│  ┌──────────────┐  ┌──────────────────────────────┐│
│  │ Local Tools   │  │ Full Tool Schemas (31 tools) ││
│  │ • memory     │  │ • Embedded at compile time   ││
│  │ • todo       │  │ • LLM sees all of them       ││
│  └──────┬───────┘  └──────────┬───────────────────┘│
│         │                     │                     │
│         │  ┌──────────────────┴──────────────┐     │
│         │  │ Agent Loop                       │     │
│         │  │ • Parses tool calls             │     │
│         │  │ • Routes local → local dispatch  │     │
│         │  │ • Routes remote → host via proto │     │
│         │  └─────────────────────────────────┘     │
└─────────┼───────────────────┼───────────────────────┘
          │                   │ stdin/stdout JSON
          │                   │
┌─────────┴───────────────────┴───────────────────────┐
│  Host Process (host_runner.py)                       │
│                                                     │
│  ┌──────────────┐  ┌──────────────────────────────┐│
│  │ LLM Client   │  │ ToolDispatcher               ││
│  │ (API key     │  │ • Imports hermes-agent tools  ││
│  │  stays here) │  │ • Dispatches to real handlers ││
│  └──────────────┘  │ • terminal → subprocess       ││
│                    │ • web_search → Firecrawl      ││
│                    │ • read_file → filesystem       ││
│                    │ • browser → Playwright         ││
│                    └──────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

## Performance Impact

- Binary size: 26MB → 26MB (tool schemas are tiny JSON)
- Cold start: unchanged (~110ms wasmtime JIT)
- Per-tool-call overhead: 1 JSON round-trip over stdin/stdout (~0.1ms)
- Host dispatch overhead: same as native hermes-agent tool execution
