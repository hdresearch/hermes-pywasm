# hermes-pyodide vs hermes-pywasm — Comparison

Two approaches to running hermes-agent as WebAssembly.

## TL;DR

| | hermes-pyodide | hermes-pywasm |
|---|---|---|
| **Approach** | Monkey-patch existing code | Ground-up rewrite of core loop |
| **Compilation** | CPython interpreter shipped as Wasm (Emscripten) | Python → C → Wasm (Nuitka AOT) |
| **Target** | Browser | WASI runtimes (wasmtime, wasmer) |
| **Binary size** | ~15-20MB (Pyodide core) + packages | 26MB (single .wasm, all-inclusive) |
| **Startup time** | 2-5 seconds (load interpreter + packages) | <100ms (native Wasm startup) |
| **New code written** | ~1,150 lines (patches + shims) | ~2,000 lines (full rewrite) |
| **Code reuse** | High — patches 8 spots in original ~12K LOC codebase | Low — rewrote from scratch using only stdlib |
| **Feature coverage** | Near-complete (same code, shimmed) | Core loop only (conversation, memory, todo) |
| **Working tools** | 8 natively + all remote via host | 2 local + all remote via host |

## Architecture

### hermes-pyodide: Shim & Patch

```
┌─── Browser ──────────────────────────────────────┐
│  Pyodide (CPython 3.11 as Wasm)                  │
│    ├── pyodide_shims.py (monkey-patches)          │
│    ├── run_agent.py (8 lines patched)             │
│    ├── agent/display.py (KawaiiSpinner → no-op)   │
│    ├── model_tools.py (3 lines patched)           │
│    ├── openai → httpx → pyfetch → browser fetch() │
│    ├── sqlite3 → wa-sqlite                        │
│    └── pydantic, yaml, jinja2 (Pyodide packages)  │
└──────────────────────────────────────────────────┘
```

- The **same run_agent.py code** runs in Pyodide
- 8 surgical patches (threading → sync, WAL → DELETE, etc.)
- `pyodide_shims.py` (294 lines) does most of the work
- HTTP goes through browser's `fetch()` via a custom httpx transport
- C extensions (pydantic-core, sqlite3) are pre-compiled Pyodide packages

### hermes-pywasm: Host-Call Protocol

```
┌─── Host (any language) ───┐     JSON lines      ┌─── Wasm sandbox ──────────┐
│  host_runner.py            │ ◄── stdin/stdout ─► │  hermes_agent.wasm         │
│  • HTTP (openai/httpx)     │                     │  • Conversation loop       │
│  • User I/O                │                     │  • Prompt building         │
│  • File persistence        │                     │  • Tool parsing/dispatch   │
│  • Remote tool execution   │                     │  • Memory/todo (local)     │
│  • Context compression     │                     │  • Token estimation        │
└────────────────────────────┘                     └────────────────────────────┘
```

- The core loop is **rewritten from scratch** using only Python stdlib
- No C extensions, no networking, no threads
- All I/O delegated to host via stdin/stdout JSON protocol
- Host can be Python, Node.js, Go, Rust, or any language that can spawn a process
- 26MB self-contained `.wasm` binary (CPython 3.11 runtime + compiled agent code)

## Detailed Comparison

### What each approach can do

| Capability | hermes-pyodide | hermes-pywasm |
|---|---|---|
| **Full conversation loop** | ✅ Same code as original | ✅ Reimplemented |
| **System prompt building** | ✅ Original prompt_builder.py | ✅ Simplified version |
| **Tool call parsing** | ✅ Original code | ✅ Reimplemented |
| **Memory tool** | ✅ Original (file-backed) | ✅ Reimplemented (protocol-backed) |
| **Todo tool** | ✅ Original | ✅ Reimplemented |
| **Clarify tool** | ✅ Original (via callback) | ✅ Via host protocol |
| **Session search (FTS5)** | ✅ wa-sqlite | ❌ No sqlite3 in WASI |
| **Session DB persistence** | ✅ SQLite + wa-sqlite | ❌ JSON state via host |
| **Context compression** | ✅ Original (LLM call via fetch) | ✅ Via host delegation |
| **Prompt caching (Claude)** | ✅ Original code | ❌ Not implemented |
| **Streaming responses** | ⚠️ With workarounds | ❌ Full response only |
| **pydantic validation** | ✅ pydantic-core Wasm package | ❌ Pure dict validation |
| **YAML config** | ✅ PyYAML Wasm package | ❌ JSON only |
| **Skills tool** | ✅ Original | ⚠️ Schema only, host provides data |
| **Remote tools** | Via host fetch() | Via host subprocess |
| **Token counting** | ⚠️ Rough (no tiktoken) | ⚠️ Rough (chars/4) |

### Developer experience

| Dimension | hermes-pyodide | hermes-pywasm |
|---|---|---|
| **Setup complexity** | `apply_patches.py` (one command) | `./build.sh` (one command) |
| **Debug cycle** | Reload browser page | Re-run Python subprocess |
| **Test without Wasm** | `python test_native.py --simulate-pyodide` | `python tests/test_protocol.py` |
| **Upstream sync** | Re-run `apply_patches.py` — may need patch updates | Manual sync of any new features |
| **Adding a new local tool** | Just import it (if pure Python + Pyodide-compatible) | Add schema + handler to `wasm_tools.py` |
| **Adding a new remote tool** | Configure in toolsets | Add schema to registry, host implements dispatch |

### Runtime characteristics

| Metric | hermes-pyodide | hermes-pywasm |
|---|---|---|
| **Cold start** | 2-5 seconds (load Pyodide + packages) | <100ms (wasmtime JIT) |
| **Memory footprint** | ~100-200MB (CPython + packages) | ~50-80MB (CPython runtime embedded) |
| **Per-turn overhead** | Minimal (interpreted Python) | Minimal (AOT-compiled Python) |
| **Sandboxing** | Browser sandbox (same-origin, CSP) | WASI sandbox (capability-based, no ambient authority) |
| **Host requirements** | Modern browser with Wasm support | wasmtime/wasmer CLI or library |
| **Can run in browser** | ✅ Yes (primary target) | ❌ No (WASI, not Emscripten) |
| **Can run on server** | ⚠️ Via Node.js + Pyodide | ✅ Yes (primary target) |
| **Can embed in other languages** | ❌ Browser JS only | ✅ Any Wasm runtime (Rust, Go, C, etc.) |

### Security model

| Aspect | hermes-pyodide | hermes-pywasm |
|---|---|---|
| **Network access** | Browser-controlled (CORS, CSP) | None (host mediates all HTTP) |
| **File access** | Emscripten virtual FS only | WASI capability-based (explicit dir grants) |
| **Subprocess** | Impossible | Impossible |
| **Secrets exposure** | API key in browser JS (visible to user) | API key in host process only (never enters Wasm) |
| **Code execution** | Full Python interpreter available | Only compiled agent code runs |

## When to use which

### Use hermes-pyodide when:
- You want a **browser-based** chat UI
- You need **maximum feature parity** with the original agent
- You want to **reuse** the existing codebase with minimal changes
- You need **C extension packages** (pydantic, sqlite3, numpy)
- Development speed matters more than startup time

### Use hermes-pywasm when:
- You want to **embed** the agent in a non-Python host (Rust, Go, edge runtime)
- **Startup time** matters (serverless, edge functions, CLI tools)
- You want **strong sandboxing** (the agent can't access anything the host doesn't explicitly provide)
- **API key security** matters (key never enters the Wasm sandbox)
- You want a **single portable binary** that runs anywhere with a Wasm runtime
- You're building a **Wasm plugin system** where the agent is one of many sandboxed modules

### Neither is appropriate when:
- You need the full 45-tool suite running locally (use native Python)
- You need real-time streaming in a CLI (use native Python)
- You need GPU access for local inference (use native Python)

## Code size

| Project | New code | Patched original code |
|---|---|---|
| hermes-pyodide | 1,147 lines | ~130 lines changed in 12K LOC codebase |
| hermes-pywasm | 2,009 lines | 0 (standalone rewrite) |
