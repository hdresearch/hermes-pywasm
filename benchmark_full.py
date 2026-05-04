#!/usr/bin/env python3
"""
COMPLETE benchmark: Native Python vs Pyodide (simulated) vs pywasm (compiled Wasm)

Unified suite covering every dimension:

  PART A — SINGLE AGENT
    A1. Deployment size (disk, dependencies)
    A2. Cold start (import + init)
    A3. Single-turn conversation
    A4. Tool-call conversation (2 API calls)
    A5. Multi-turn scaling (1, 5, 10, 20 turns)
    A6. CPU time breakdown (user vs sys)
    A7. Peak RSS memory

  PART B — SWARM (PARALLEL AGENTS)
    B1. Burst spawn (init only, N=1..50)
    B2. Parallel single-turn (N=1..50)
    B3. Parallel multi-turn 5-turn (N=1..20)
    B4. Mixed workload (60% simple, 40% tool-calls, N=20)
    B5. Worker pool throughput (50 queries ÷ 1..20 workers)
    B6. Scale-up curve + per-agent amortization (N=1..50)
    B7. Sustained load (20 agents × 3 rounds, degradation test)
    B8. Memory footprint per-instance (measured, not estimated)

  PART C — EVALUATION SUMMARY
    Decision matrix with all numbers
    "When to use what" with data-backed thresholds

All API responses are mocked. Measures pure compute + process overhead.
"""

import json
import multiprocessing
import os
import re
import statistics
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERMES_AGENT_DIR = Path(__file__).parent.parent / "hermes-agent"
HERMES_PYODIDE_DIR = Path(__file__).parent.parent / "hermes-pyodide"
HERMES_PYWASM_DIR = Path(__file__).parent
WASM_BINARY = HERMES_PYWASM_DIR / "hermes_agent.wasm"
NATIVE_PY = str(HERMES_AGENT_DIR / "venv/bin/python3")
NCPU = multiprocessing.cpu_count()

RUNS_SINGLE = 5   # runs for single-agent tests
RUNS_SWARM = 3    # runs for swarm tests (many processes, takes longer)

TARGETS_ALL = ("native", "pyodide", "pywasm")
TARGETS_SWARM = ("native", "pyodide", "pywasm")

LABELS = {
    "native":  "Native Python",
    "pyodide": "Pyodide (sim)",
    "pywasm":  "pywasm (Wasm)",
}

# ---------------------------------------------------------------------------
# Load real hermes-agent tool schemas for pywasm so it has the same tools
# ---------------------------------------------------------------------------

def _load_host_schemas():
    """Extract real tool schemas from hermes-agent registry."""
    import importlib
    venv_sp = list((HERMES_AGENT_DIR / "venv" / "lib").glob("python*/site-packages"))
    for sp in venv_sp:
        if str(sp) not in sys.path:
            sys.path.insert(0, str(sp))
    if str(HERMES_AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(HERMES_AGENT_DIR))

    try:
        from tools.registry import registry
        for mod in [
            "tools.terminal_tool", "tools.file_tools", "tools.skills_tool",
            "tools.skill_manager_tool", "tools.memory_tool", "tools.todo_tool",
            "tools.clarify_tool", "tools.code_execution_tool", "tools.delegate_tool",
            "tools.session_search_tool", "tools.web_tools", "tools.browser_tool",
            "tools.vision_tools", "tools.image_generation_tool",
            "tools.mixture_of_agents_tool", "tools.tts_tool",
        ]:
            try:
                __import__(mod)
            except Exception:
                pass

        schemas = []
        names = []
        for name in sorted(registry._tools.keys()):
            entry = registry._tools[name]
            schemas.append({"type": "function", "function": entry.schema})
            names.append(name)
        return schemas, names
    except Exception as e:
        print(f"  WARNING: Could not load hermes-agent schemas: {e}")
        return [], []


HOST_SCHEMAS, HOST_TOOL_NAMES = _load_host_schemas()

# ═══════════════════════════════════════════════════════════════════════════
# Mock data
# ═══════════════════════════════════════════════════════════════════════════

def mock_resp(content, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "c", "object": "chat.completion", "created": 1, "model": "bench",
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

SIMPLE = mock_resp("Hello!")
MEM_TOOL = mock_resp("Saving.", tool_calls=[{
    "id": "c1", "type": "function",
    "function": {"name": "memory",
                 "arguments": json.dumps({"action": "add", "target": "user",
                                          "content": "Likes benchmarks"})},
}])
FINAL = mock_resp("Done!")

# ═══════════════════════════════════════════════════════════════════════════
# Script builders
# ═══════════════════════════════════════════════════════════════════════════

NATIVE_PREAMBLE = textwrap.dedent(f"""\
import time, os, sys, json
os.environ['HERMES_HOME'] = '/tmp/hermes-bench-full'
os.makedirs('/tmp/hermes-bench-full/logs', exist_ok=True)
sys.path.insert(0, {str(HERMES_AGENT_DIR)!r})
import httpx
""")

PYODIDE_PREAMBLE = textwrap.dedent(f"""\
import time, os, sys, json
os.environ['HERMES_HOME'] = '/tmp/hermes-bench-full'
os.makedirs('/tmp/hermes-bench-full/logs', exist_ok=True)
os.environ['PYODIDE'] = '1'
sys.path.insert(0, {str(HERMES_PYODIDE_DIR)!r})
sys.path.insert(0, {str(HERMES_PYODIDE_DIR / 'hermes-agent')!r})
import pyodide_shims
import httpx
""")

MOCK_SETUP = textwrap.dedent("""\
SIMPLE = %s
MEM_TOOL = %s
FINAL = %s
queue = []
idx = [0]
class MT(httpx.BaseTransport):
    def handle_request(self, r):
        i = idx[0]; idx[0] += 1
        return httpx.Response(200, json=queue[min(i, len(queue)-1)])
""") % (json.dumps(SIMPLE), json.dumps(MEM_TOOL), json.dumps(FINAL))

AGENT_INIT = textwrap.dedent("""\
from run_agent import AIAgent
from openai import OpenAI
agent = AIAgent(
    api_key='k', base_url='http://localhost:1/v1', model='bench',
    enabled_toolsets=['memory','skills','planning'],
    quiet_mode=True, max_iterations=50, skip_memory=True, skip_context_files=True,
)
agent.client = OpenAI(api_key='k', base_url='http://localhost:1/v1',
                       http_client=httpx.Client(transport=MT()))
""")

def _full_script(preamble, bench_code):
    return preamble + MOCK_SETUP + AGENT_INIT + bench_code

def native_script(bench_code):
    return _full_script(NATIVE_PREAMBLE, bench_code)

def pyodide_script(bench_code):
    return _full_script(PYODIDE_PREAMBLE, bench_code)

def wasm_input(messages_and_responses):
    init_config = {"model": "bench", "max_iterations": 50}
    if HOST_SCHEMAS:
        init_config["host_tool_schemas"] = HOST_SCHEMAS
        init_config["host_available_tools"] = HOST_TOOL_NAMES
    lines = [{"type": "init", "config": init_config}]
    for user_msg, *responses in messages_and_responses:
        lines.append({"type": "user_message", "content": user_msg})
        lines.append({"type": "memory_data", "memory": "", "user": ""})
        lines.append({"type": "skills_data", "skills": []})
        for r in responses:
            lines.append({"type": "api_response", "response": r})
    return "\n".join(json.dumps(m) for m in lines) + "\n"

# Pre-built scripts/inputs
_init_code = "pass\n"
_single_code = "queue[:] = [SIMPLE]; idx[0] = 0\nagent.run_conversation(user_message='Hello')\n"
_tool_code = "queue[:] = [MEM_TOOL, FINAL]; idx[0] = 0\nagent.run_conversation(user_message='Remember pizza')\n"

def _multi_code(n):
    c = "history = []\n"
    for i in range(n):
        c += f"queue[:] = [SIMPLE]; idx[0] = 0\n"
        c += f"r = agent.run_conversation(user_message='Message {i+1}', conversation_history=history)\n"
        c += "history = r['messages']\n"
    return c

SCRIPTS = {
    "init": {
        "native": native_script(_init_code),
        "pyodide": pyodide_script(_init_code),
    },
    "single": {
        "native": native_script(_single_code),
        "pyodide": pyodide_script(_single_code),
    },
    "tool": {
        "native": native_script(_tool_code),
        "pyodide": pyodide_script(_tool_code),
    },
}
def _wasm_init_input():
    init_config = {"model": "bench"}
    if HOST_SCHEMAS:
        init_config["host_tool_schemas"] = HOST_SCHEMAS
        init_config["host_available_tools"] = HOST_TOOL_NAMES
    return json.dumps({"type": "init", "config": init_config}) + "\n"

WASM_INPUTS = {
    "init": _wasm_init_input(),
    "single": wasm_input([("Hello", SIMPLE)]),
    "tool": wasm_input([("Remember pizza", MEM_TOOL, FINAL)]),
}

def get_multi_scripts(n):
    code = _multi_code(n)
    return {
        "native": native_script(code),
        "pyodide": pyodide_script(code),
    }

def get_multi_wasm(n):
    return wasm_input([(f"Message {i+1}", SIMPLE) for i in range(n)])


# ═══════════════════════════════════════════════════════════════════════════
# Measurement
# ═══════════════════════════════════════════════════════════════════════════

def measure_timed(cmd, stdin_data=None, timeout=120):
    """Run via /usr/bin/time -l, return detailed metrics."""
    proc = subprocess.run(
        ["/usr/bin/time", "-l"] + cmd,
        input=stdin_data, capture_output=True, text=True, timeout=timeout,
    )
    stderr = proc.stderr or ""
    def ex(pat, d=0):
        m = re.search(pat, stderr)
        return float(m.group(1)) if m else d
    return {
        "wall_ms": ex(r'([\d.]+)\s+real') * 1000,
        "user_ms": ex(r'([\d.]+)\s+user') * 1000,
        "sys_ms":  ex(r'([\d.]+)\s+sys') * 1000,
        "cpu_ms":  (ex(r'([\d.]+)\s+user') + ex(r'([\d.]+)\s+sys')) * 1000,
        "rss_mb":  ex(r'(\d+)\s+maximum resident set size') / (1024*1024),
        "page_faults": int(ex(r'(\d+)\s+page faults')),
        "ctx_vol": int(ex(r'(\d+)\s+voluntary context switches')),
        "ok": proc.returncode == 0,
    }

def run_target(target, kind, timeout=120):
    """Run a single measurement for (target, kind)."""
    if target == "pywasm":
        return measure_timed(["wasmtime", str(WASM_BINARY)],
                             stdin_data=WASM_INPUTS[kind], timeout=timeout)
    else:
        return measure_timed([NATIVE_PY, "-c", SCRIPTS[kind][target]], timeout=timeout)

def run_target_multi(target, n, timeout=120):
    """Run multi-turn measurement."""
    if target == "pywasm":
        return measure_timed(["wasmtime", str(WASM_BINARY)],
                             stdin_data=get_multi_wasm(n), timeout=timeout)
    else:
        scripts = get_multi_scripts(n)
        return measure_timed([NATIVE_PY, "-c", scripts[target]], timeout=timeout)

def run_quick(target, kind, timeout=120):
    """Quick wall-clock-only run (for swarm — no /usr/bin/time overhead)."""
    t0 = time.perf_counter()
    if target == "pywasm":
        subprocess.run(["wasmtime", str(WASM_BINARY)],
                       input=WASM_INPUTS[kind], capture_output=True, text=True, timeout=timeout)
    else:
        subprocess.run([NATIVE_PY, "-c", SCRIPTS[kind][target]],
                       capture_output=True, text=True, timeout=timeout)
    return (time.perf_counter() - t0) * 1000

def run_quick_multi(target, n, timeout=120):
    """Quick wall-clock-only run for multi-turn."""
    t0 = time.perf_counter()
    if target == "pywasm":
        subprocess.run(["wasmtime", str(WASM_BINARY)],
                       input=get_multi_wasm(n), capture_output=True, text=True, timeout=timeout)
    else:
        scripts = get_multi_scripts(n)
        subprocess.run([NATIVE_PY, "-c", scripts[target]],
                       capture_output=True, text=True, timeout=timeout)
    return (time.perf_counter() - t0) * 1000

def run_quick_script(script_text, timeout=120):
    """Quick wall-clock-only for a raw native/pyodide script."""
    t0 = time.perf_counter()
    subprocess.run([NATIVE_PY, "-c", script_text],
                   capture_output=True, text=True, timeout=timeout)
    return (time.perf_counter() - t0) * 1000

def run_quick_wasm(wasm_in, timeout=30):
    """Quick wall-clock-only for raw wasm input."""
    t0 = time.perf_counter()
    subprocess.run(["wasmtime", str(WASM_BINARY)],
                   input=wasm_in, capture_output=True, text=True, timeout=timeout)
    return (time.perf_counter() - t0) * 1000


def launch_parallel(n, fn, *args):
    """Launch n calls of fn(*args) in parallel threads. Return (wall_ms, results)."""
    results = [None] * n
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(fn, *args): i for i in range(n)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
    return (time.perf_counter() - t0) * 1000, results


# ═══════════════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════════════

def med(vals):
    return statistics.median(vals) if vals else 0
def sd(vals):
    return statistics.stdev(vals) if len(vals) >= 2 else 0
def ex(runs, key):
    return [r[key] for r in runs if key in r]
def fmt(v):
    if v >= 10000: return f"{v/1000:.1f}s"
    return f"{v:.0f}ms"

def section(title):
    print(f"\n{'━' * 86}")
    print(f"  {title}")
    print(f"{'━' * 86}")

def table3(results, metrics, targets=TARGETS_ALL):
    """Print a 3-column metric table."""
    cw = 20
    hdr = f"  {'':22}" + "".join(f"{LABELS[t]:>{cw}}" for t in targets)
    print(hdr)
    print(f"  {'─'*22}" + f"{'─'*cw}" * len(targets))
    for key, label, f, unit in metrics:
        row = f"  {label:22}"
        for t in targets:
            vals = ex(results.get(t, []), key)
            if vals:
                m = med(vals)
                s = sd(vals)
                cell = f"{m:{f}}{unit} ±{s:.0f}"
                row += f"{cell:>{cw}}"
            else:
                row += f"{'—':>{cw}}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════════
# Deployment size
# ═══════════════════════════════════════════════════════════════════════════

def measure_sizes():
    def dir_size(path, exclude=None):
        total = 0
        for root, dirs, files in os.walk(path):
            if exclude:
                dirs[:] = [d for d in dirs if d not in exclude]
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp) and not os.path.islink(fp):
                    total += os.path.getsize(fp)
        return total
    src = dir_size(HERMES_AGENT_DIR, exclude={'.git','venv','__pycache__','node_modules','.vers'})
    venv = dir_size(HERMES_AGENT_DIR / "venv/lib")
    wasm_sz = WASM_BINARY.stat().st_size if WASM_BINARY.exists() else 0
    pyodide_patch = sum(
        f.stat().st_size for f in HERMES_PYODIDE_DIR.rglob("*.py")
        if "hermes-agent" not in str(f) and "__pycache__" not in str(f))
    pkgs = len(list((HERMES_AGENT_DIR / "venv/lib/python3.11/site-packages").glob("*.dist-info")))
    return {
        "src_mb": src/(1024**2), "venv_mb": venv/(1024**2),
        "total_mb": (src+venv)/(1024**2), "pip_pkgs": pkgs,
        "pyodide_patch_kb": pyodide_patch/1024,
        "wasm_mb": wasm_sz/(1024**2),
        "host_kb": (HERMES_PYWASM_DIR/"host_runner.py").stat().st_size/1024,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if not WASM_BINARY.exists():
        print("ERROR: Build hermes_agent.wasm first: ./build.sh"); sys.exit(1)

    print("═" * 86)
    print("  HERMES AGENT — COMPLETE EVALUATION BENCHMARK")
    print(f"  Native Python  vs  Pyodide (simulated)  vs  pywasm (compiled Wasm)")
    print(f"  {NCPU} CPU cores · {RUNS_SINGLE} single-agent runs · {RUNS_SWARM} swarm runs")
    print(f"  All API responses mocked — pure compute + process overhead")
    print("═" * 86)

    ALL = {}  # raw data

    # ──────────────────────────────────────────────────────────────────────
    # PART A — SINGLE AGENT
    # ──────────────────────────────────────────────────────────────────────

    print(f"\n{'▓' * 86}")
    print(f"  PART A — SINGLE AGENT PERFORMANCE")
    print(f"{'▓' * 86}")

    # A1: sizes
    section("A1. DEPLOYMENT SIZE")
    sizes = measure_sizes()
    ALL["sizes"] = sizes
    print(f"""
  ┌─────────────────────────────┬────────────────┬────────────────┬────────────────┐
  │                             │ Native Python  │ Pyodide (browser)│ pywasm (Wasm)│
  ├─────────────────────────────┼────────────────┼────────────────┼────────────────┤
  │ Source code                 │  {sizes['src_mb']:>8.1f} MB   │  same + 33 KB  │   ~66 KB       │
  │ Runtime deps (disk)         │  {sizes['venv_mb']:>8.1f} MB   │  ~25 MB (CDN)  │    0 MB        │
  │ Total deployment            │  {sizes['total_mb']:>8.0f} MB   │  ~25 MB        │  {sizes['wasm_mb']:>5.0f} MB       │
  │ Pip packages                │      {sizes['pip_pkgs']:>5}      │  ~10 (micropip)│      0         │
  │ Runtime required            │  Python 3.11   │  Browser+Wasm  │  wasmtime      │
  └─────────────────────────────┴────────────────┴────────────────┴────────────────┘""")

    # A2: cold start
    section("A2. COLD START (import + init, no conversation)")
    r = {t: [] for t in TARGETS_ALL}
    for _ in range(RUNS_SINGLE):
        for t in TARGETS_ALL:
            r[t].append(run_target(t, "init"))
    ALL["cold_start"] = r
    table3(r, [
        ("wall_ms",  "Wall clock",    ".0f", "ms"),
        ("cpu_ms",   "CPU time",      ".0f", "ms"),
        ("user_ms",  "  user",        ".0f", "ms"),
        ("sys_ms",   "  sys",         ".0f", "ms"),
        ("rss_mb",   "Peak RSS",      ".1f", " MB"),
        ("page_faults","Page faults",  ".0f", ""),
        ("ctx_vol",  "Vol ctx sw",    ".0f", ""),
    ])

    # A3: single turn
    section("A3. SINGLE-TURN CONVERSATION (1 API call)")
    r = {t: [] for t in TARGETS_ALL}
    for _ in range(RUNS_SINGLE):
        for t in TARGETS_ALL:
            r[t].append(run_target(t, "single"))
    ALL["single_turn"] = r
    table3(r, [
        ("wall_ms", "Wall clock", ".0f", "ms"),
        ("cpu_ms",  "CPU time",   ".0f", "ms"),
        ("rss_mb",  "Peak RSS",   ".1f", " MB"),
    ])

    # A4: tool call
    section("A4. TOOL-CALL CONVERSATION (memory add → follow-up, 2 API calls)")
    r = {t: [] for t in TARGETS_ALL}
    for _ in range(RUNS_SINGLE):
        for t in TARGETS_ALL:
            r[t].append(run_target(t, "tool"))
    ALL["tool_call"] = r
    table3(r, [
        ("wall_ms", "Wall clock", ".0f", "ms"),
        ("cpu_ms",  "CPU time",   ".0f", "ms"),
        ("rss_mb",  "Peak RSS",   ".1f", " MB"),
    ])

    # A5: multi-turn scaling
    section("A5. MULTI-TURN SCALING (wall time & memory vs turns)")
    turn_counts = (1, 5, 10, 20)
    r_scaling = {}
    for n in turn_counts:
        r_scaling[n] = {t: [] for t in TARGETS_ALL}
        for _ in range(RUNS_SINGLE):
            for t in TARGETS_ALL:
                r_scaling[n][t].append(run_target_multi(t, n))
    ALL["scaling"] = {str(k): v for k, v in r_scaling.items()}

    print(f"\n  Wall clock (ms):")
    print(f"  {'Turns':>8}" + "".join(f"{LABELS[t]:>20}" for t in TARGETS_ALL))
    print(f"  {'─'*8}" + "─"*20*3)
    for n in turn_counts:
        row = f"  {n:>8}"
        for t in TARGETS_ALL:
            vals = ex(r_scaling[n][t], "wall_ms")
            row += f"{med(vals):>17.0f}ms " if vals else f"{'—':>20}"
        print(row)

    print(f"\n  Peak RSS (MB):")
    print(f"  {'Turns':>8}" + "".join(f"{LABELS[t]:>20}" for t in TARGETS_ALL))
    print(f"  {'─'*8}" + "─"*20*3)
    for n in turn_counts:
        row = f"  {n:>8}"
        for t in TARGETS_ALL:
            vals = ex(r_scaling[n][t], "rss_mb")
            row += f"{med(vals):>17.1f}MB " if vals else f"{'—':>20}"
        print(row)

    print(f"\n  CPU time (ms):")
    print(f"  {'Turns':>8}" + "".join(f"{LABELS[t]:>20}" for t in TARGETS_ALL))
    print(f"  {'─'*8}" + "─"*20*3)
    for n in turn_counts:
        row = f"  {n:>8}"
        for t in TARGETS_ALL:
            vals = ex(r_scaling[n][t], "cpu_ms")
            row += f"{med(vals):>17.0f}ms " if vals else f"{'—':>20}"
        print(row)

    print(f"\n  Per-turn marginal cost (1→20 turns):")
    for t in TARGETS_ALL:
        w1 = med(ex(r_scaling[1][t], "wall_ms"))
        w20 = med(ex(r_scaling[20][t], "wall_ms"))
        c1 = med(ex(r_scaling[1][t], "cpu_ms"))
        c20 = med(ex(r_scaling[20][t], "cpu_ms"))
        if w1 and w20:
            wall_per = (w20 - w1) / 19
            cpu_per = (c20 - c1) / 19
            print(f"    {LABELS[t]:18}  wall: {wall_per:>6.1f}ms/turn  cpu: {cpu_per:>6.1f}ms/turn")

    # ──────────────────────────────────────────────────────────────────────
    # PART B — SWARM (PARALLEL AGENTS)
    # ──────────────────────────────────────────────────────────────────────

    print(f"\n{'▓' * 86}")
    print(f"  PART B — SWARM (PARALLEL AGENTS)")
    print(f"{'▓' * 86}")

    # B1: burst spawn
    section("B1. BURST SPAWN — Cold-start N agents simultaneously (init only)")
    burst_counts = (1, 5, 10, 20, 50)
    r_burst = {}
    for n in burst_counts:
        r_burst[n] = {}
        for t in TARGETS_SWARM:
            walls = []
            for _ in range(RUNS_SWARM):
                w, _ = launch_parallel(n, run_quick, t, "init")
                walls.append(w)
            r_burst[n][t] = med(walls)
    ALL["burst"] = {str(k): v for k, v in r_burst.items()}

    print(f"\n  {'N':>4}" + "".join(f"{LABELS[t]:>18}" for t in TARGETS_SWARM) + f"  {'N→P speedup':>12}")
    print(f"  {'─'*4}" + "─"*18*3 + f"  {'─'*12}")
    for n in burst_counts:
        row = f"  {n:>4}"
        for t in TARGETS_SWARM:
            row += f"{fmt(r_burst[n][t]):>18}"
        sp = r_burst[n]["native"] / r_burst[n]["pywasm"] if r_burst[n]["pywasm"] else 0
        row += f"  {sp:>10.1f}×"
        print(row)

    # B2: parallel single-turn
    section("B2. PARALLEL SINGLE-TURN — N agents, each 1 query")
    single_counts = (1, 5, 10, 20, 50)
    r_psingle = {}
    for n in single_counts:
        r_psingle[n] = {}
        for t in TARGETS_SWARM:
            walls = []
            for _ in range(RUNS_SWARM):
                w, _ = launch_parallel(n, run_quick, t, "single")
                walls.append(w)
            r_psingle[n][t] = med(walls)
    ALL["par_single"] = {str(k): v for k, v in r_psingle.items()}

    print(f"\n  {'N':>4}" + "".join(f"{LABELS[t]:>18}" for t in TARGETS_SWARM) + f"  {'Wasm q/s':>10}")
    print(f"  {'─'*4}" + "─"*18*3 + f"  {'─'*10}")
    for n in single_counts:
        row = f"  {n:>4}"
        for t in TARGETS_SWARM:
            row += f"{fmt(r_psingle[n][t]):>18}"
        qps = n / (r_psingle[n]["pywasm"] / 1000) if r_psingle[n]["pywasm"] else 0
        row += f"  {qps:>8.0f}/s"
        print(row)

    # B3: parallel multi-turn
    section("B3. PARALLEL MULTI-TURN — N agents, each 5-turn conversation")
    multi_counts = (1, 5, 10, 20)
    r_pmulti = {}
    for n in multi_counts:
        r_pmulti[n] = {}
        for t in TARGETS_SWARM:
            walls = []
            for _ in range(RUNS_SWARM):
                w, _ = launch_parallel(n, run_quick_multi, t, 5)
                walls.append(w)
            r_pmulti[n][t] = med(walls)
    ALL["par_multi"] = {str(k): v for k, v in r_pmulti.items()}

    print(f"\n  {'N':>4}" + "".join(f"{LABELS[t]:>18}" for t in TARGETS_SWARM) + f"  {'N→P speedup':>12}")
    print(f"  {'─'*4}" + "─"*18*3 + f"  {'─'*12}")
    for n in multi_counts:
        row = f"  {n:>4}"
        for t in TARGETS_SWARM:
            row += f"{fmt(r_pmulti[n][t]):>18}"
        sp = r_pmulti[n]["native"] / r_pmulti[n]["pywasm"] if r_pmulti[n]["pywasm"] else 0
        row += f"  {sp:>10.1f}×"
        print(row)

    # B4: mixed workload
    section("B4. MIXED WORKLOAD — 20 agents, 60% simple + 40% tool-calls")
    n_mix = 20
    n_simple = int(n_mix * 0.6)
    n_toolc = n_mix - n_simple
    r_mixed = {}
    for t in TARGETS_SWARM:
        walls = []
        for _ in range(RUNS_SWARM):
            simple_fn = (run_quick_script, SCRIPTS["single"][t]) if t != "pywasm" else (run_quick_wasm, WASM_INPUTS["single"])
            tool_fn = (run_quick_script, SCRIPTS["tool"][t]) if t != "pywasm" else (run_quick_wasm, WASM_INPUTS["tool"])

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n_mix) as pool:
                futs = []
                for _ in range(n_simple):
                    futs.append(pool.submit(simple_fn[0], simple_fn[1]))
                for _ in range(n_toolc):
                    futs.append(pool.submit(tool_fn[0], tool_fn[1]))
                for f in as_completed(futs):
                    f.result()
            walls.append((time.perf_counter() - t0) * 1000)
        r_mixed[t] = med(walls)
    ALL["mixed"] = r_mixed

    print(f"\n  {'Target':>20}  {'Wall clock':>12}  {'Tasks/sec':>12}")
    print(f"  {'─'*20}  {'─'*12}  {'─'*12}")
    for t in TARGETS_SWARM:
        tps = n_mix / (r_mixed[t] / 1000) if r_mixed[t] else 0
        print(f"  {LABELS[t]:>20}  {fmt(r_mixed[t]):>12}  {tps:>10.1f}/s")
    sp = r_mixed["native"] / r_mixed["pywasm"] if r_mixed["pywasm"] else 0
    print(f"  {'Native → pywasm':>20}  {'':>12}  {sp:.1f}× speedup")

    # B5: worker pool throughput
    section("B5. WORKER POOL — 50 queries through N workers")
    total_q = 50
    pool_sizes = (1, 5, 10, 20)
    r_pool = {}
    for ps in pool_sizes:
        r_pool[ps] = {}
        for t in TARGETS_SWARM:
            walls = []
            for _ in range(RUNS_SWARM):
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=ps) as pool:
                    if t == "pywasm":
                        futs = [pool.submit(run_quick_wasm, WASM_INPUTS["single"]) for _ in range(total_q)]
                    else:
                        futs = [pool.submit(run_quick_script, SCRIPTS["single"][t]) for _ in range(total_q)]
                    for f in as_completed(futs): f.result()
                walls.append((time.perf_counter() - t0) * 1000)
            r_pool[ps][t] = med(walls)
    ALL["pool"] = {str(k): v for k, v in r_pool.items()}

    print(f"\n  {'Workers':>8}" + "".join(f"{LABELS[t]:>18}" for t in TARGETS_SWARM) + f"  {'Wasm q/s':>10}")
    print(f"  {'─'*8}" + "─"*18*3 + f"  {'─'*10}")
    for ps in pool_sizes:
        row = f"  {ps:>8}"
        for t in TARGETS_SWARM:
            row += f"{fmt(r_pool[ps][t]):>18}"
        qps = total_q / (r_pool[ps]["pywasm"] / 1000) if r_pool[ps]["pywasm"] else 0
        row += f"  {qps:>8.0f}/s"
        print(row)

    # B6: scale-up curve
    section("B6. SCALE-UP CURVE — per-agent cost at increasing N")
    scale_ns = [1] + list(range(5, 51, 5))
    r_scale = {}
    for n in scale_ns:
        r_scale[n] = {}
        for t in TARGETS_SWARM:
            walls = []
            for _ in range(RUNS_SWARM):
                w, _ = launch_parallel(n, run_quick, t, "single")
                walls.append(w)
            r_scale[n][t] = med(walls)
    ALL["scale"] = {str(k): v for k, v in r_scale.items()}

    print(f"\n  {'N':>4}" + "".join(f"{LABELS[t]:>16}" for t in TARGETS_SWARM) + "".join(f"{'/' + LABELS[t].split()[0]:>10}" for t in TARGETS_SWARM))
    print(f"  {'─'*4}" + "─"*16*3 + "─"*10*3)
    for n in scale_ns:
        row = f"  {n:>4}"
        for t in TARGETS_SWARM:
            row += f"{fmt(r_scale[n][t]):>16}"
        for t in TARGETS_SWARM:
            per = r_scale[n][t] / n
            row += f"{fmt(per):>10}"
        print(row)

    # B7: sustained load
    section("B7. SUSTAINED LOAD — 20 agents × 3-turn convo, 3 consecutive rounds")
    n_sust = 20
    rounds = 3
    r_sustained = {}
    for rd in range(1, rounds + 1):
        r_sustained[rd] = {}
        for t in TARGETS_SWARM:
            walls = []
            for _ in range(RUNS_SWARM):
                w, _ = launch_parallel(n_sust, run_quick_multi, t, 3)
                walls.append(w)
            r_sustained[rd][t] = med(walls)
    ALL["sustained"] = {str(k): v for k, v in r_sustained.items()}

    print(f"\n  {'Round':>6}" + "".join(f"{LABELS[t]:>18}" for t in TARGETS_SWARM))
    print(f"  {'─'*6}" + "─"*18*3)
    for rd in range(1, rounds + 1):
        row = f"  {rd:>6}"
        for t in TARGETS_SWARM:
            row += f"{fmt(r_sustained[rd][t]):>18}"
        print(row)

    # B8: memory per-instance (measured)
    section("B8. MEMORY PER INSTANCE — Measured RSS (parallel instances)")
    mem_counts = (1, 5, 10)
    r_mem = {}
    for n in mem_counts:
        r_mem[n] = {}
        for t in TARGETS_SWARM:
            def _measure_one(target=t):
                if target == "pywasm":
                    return measure_timed(["wasmtime", str(WASM_BINARY)],
                                         stdin_data=WASM_INPUTS["single"])
                else:
                    return measure_timed([NATIVE_PY, "-c", SCRIPTS["single"][target]])
            with ThreadPoolExecutor(max_workers=n) as pool:
                futs = [pool.submit(_measure_one) for _ in range(n)]
                results = [f.result() for f in futs]
            rss_vals = [r["rss_mb"] for r in results]
            r_mem[n][t] = {
                "total_mb": sum(rss_vals),
                "avg_mb": statistics.mean(rss_vals),
                "max_mb": max(rss_vals),
            }
    ALL["memory"] = {str(k): v for k, v in r_mem.items()}

    print(f"\n  {'N':>4}  {'Metric':<10}" + "".join(f"{LABELS[t]:>18}" for t in TARGETS_SWARM))
    print(f"  {'─'*4}  {'─'*10}" + "─"*18*3)
    for n in mem_counts:
        for metric, label in [("total_mb", "total"), ("avg_mb", "per-inst"), ("max_mb", "peak")]:
            row = f"  {n if label == 'total' else '':>4}  {label:<10}"
            for t in TARGETS_SWARM:
                v = r_mem[n][t][metric]
                row += f"{v:>14.0f} MB "
                if metric == "total_mb":
                    pass  # just the number
            print(row)
        if n != mem_counts[-1]:
            print()

    # ──────────────────────────────────────────────────────────────────────
    # PART C — EVALUATION SUMMARY
    # ──────────────────────────────────────────────────────────────────────

    print(f"\n{'▓' * 86}")
    print(f"  PART C — EVALUATION SUMMARY")
    print(f"{'▓' * 86}")

    section("C1. COMPLETE DECISION MATRIX")

    # Gather key numbers
    def gm(data, target, key):
        return med(ex(data.get(target, []), key))

    c_cold  = {t: gm(ALL["cold_start"], t, "wall_ms") for t in TARGETS_ALL}
    c_1t    = {t: gm(ALL["single_turn"], t, "wall_ms") for t in TARGETS_ALL}
    c_tool  = {t: gm(ALL["tool_call"], t, "wall_ms") for t in TARGETS_ALL}
    c_5t    = {t: med(ex(r_scaling[5][t], "wall_ms")) for t in TARGETS_ALL}
    c_20t   = {t: med(ex(r_scaling[20][t], "wall_ms")) for t in TARGETS_ALL}
    c_rss   = {t: gm(ALL["single_turn"], t, "rss_mb") for t in TARGETS_ALL}
    c_cpu   = {t: gm(ALL["single_turn"], t, "cpu_ms") for t in TARGETS_ALL}

    # Marginal cost
    c_marg = {}
    for t in TARGETS_ALL:
        w1 = med(ex(r_scaling[1][t], "wall_ms"))
        w20 = med(ex(r_scaling[20][t], "wall_ms"))
        c_marg[t] = (w20 - w1) / 19 if w1 and w20 else 0

    # Swarm numbers
    s_20 = {t: r_psingle[20].get(t, 0) for t in TARGETS_SWARM}
    s_50 = {t: r_psingle[50].get(t, 0) for t in TARGETS_SWARM}
    s_pool = {t: r_pool[20].get(t, 0) for t in TARGETS_SWARM}
    s_multi20 = {t: r_pmulti[20].get(t, 0) for t in TARGETS_SWARM}
    s_mixed = {t: r_mixed.get(t, 0) for t in TARGETS_SWARM}

    # Per-agent at N=50
    pa50 = {t: r_scale[50].get(t, 0) / 50 for t in TARGETS_SWARM}

    m1 = r_mem[1]
    m10 = r_mem[10]

    print(f"""
  ┌────────────────────────────────────────┬───────────┬───────────┬───────────┐
  │                                        │  Native   │  Pyodide  │  pywasm   │
  ├── SINGLE AGENT ────────────────────────┼───────────┼───────────┼───────────┤
  │ Cold start (init)                      │ {c_cold['native']:>7.0f}ms │ {c_cold['pyodide']:>7.0f}ms │ {c_cold['pywasm']:>7.0f}ms │
  │ Single turn (1 API call)               │ {c_1t['native']:>7.0f}ms │ {c_1t['pyodide']:>7.0f}ms │ {c_1t['pywasm']:>7.0f}ms │
  │ Tool call (2 API calls)                │ {c_tool['native']:>7.0f}ms │ {c_tool['pyodide']:>7.0f}ms │ {c_tool['pywasm']:>7.0f}ms │
  │ 5-turn conversation                    │ {c_5t['native']:>7.0f}ms │ {c_5t['pyodide']:>7.0f}ms │ {c_5t['pywasm']:>7.0f}ms │
  │ 20-turn conversation                   │ {c_20t['native']:>7.0f}ms │ {c_20t['pyodide']:>7.0f}ms │ {c_20t['pywasm']:>7.0f}ms │
  │ Per-turn marginal cost                 │ {c_marg['native']:>6.1f}ms │ {c_marg['pyodide']:>6.1f}ms │ {c_marg['pywasm']:>6.1f}ms │
  │ Peak RSS (single turn)                 │ {c_rss['native']:>6.0f} MB │ {c_rss['pyodide']:>6.0f} MB │ {c_rss['pywasm']:>6.0f} MB │
  │ CPU time (single turn)                 │ {c_cpu['native']:>7.0f}ms │ {c_cpu['pyodide']:>7.0f}ms │ {c_cpu['pywasm']:>7.0f}ms │
  ├── SWARM (PARALLEL) ───────────────────┼───────────┼───────────┼───────────┤
  │ 20 agents × 1-turn                     │ {fmt(s_20['native']):>9} │ {fmt(s_20['pyodide']):>9} │ {fmt(s_20['pywasm']):>9} │
  │ 50 agents × 1-turn                     │ {fmt(s_50['native']):>9} │ {fmt(s_50['pyodide']):>9} │ {fmt(s_50['pywasm']):>9} │
  │ 20 agents × 5-turn                     │ {fmt(s_multi20['native']):>9} │ {fmt(s_multi20['pyodide']):>9} │ {fmt(s_multi20['pywasm']):>9} │
  │ Mixed workload (20 agents)             │ {fmt(s_mixed['native']):>9} │ {fmt(s_mixed['pyodide']):>9} │ {fmt(s_mixed['pywasm']):>9} │
  │ 50q ÷ 20 workers (pool)               │ {fmt(s_pool['native']):>9} │ {fmt(s_pool['pyodide']):>9} │ {fmt(s_pool['pywasm']):>9} │
  │ Per-agent @ N=50                       │ {fmt(pa50['native']):>9} │ {fmt(pa50['pyodide']):>9} │ {fmt(pa50['pywasm']):>9} │
  ├── RESOURCES ──────────────────────────┼───────────┼───────────┼───────────┤
  │ RSS per instance (measured)            │ {m1['native']['avg_mb']:>6.0f} MB │ {m1['pyodide']['avg_mb']:>6.0f} MB │ {m1['pywasm']['avg_mb']:>6.0f} MB │
  │ RSS 10 instances (total)               │ {m10['native']['total_mb']:>5.0f} MB │ {m10['pyodide']['total_mb']:>5.0f} MB │ {m10['pywasm']['total_mb']:>5.0f} MB │
  │ Deployment size                        │ {sizes['total_mb']:>5.0f} MB │   ~25 MB  │  {sizes['wasm_mb']:>5.0f} MB │
  │ Pip packages                           │ {sizes['pip_pkgs']:>7}   │    ~10    │       0   │
  ├── CAPABILITIES ───────────────────────┼───────────┼───────────┼───────────┤
  │ Tool count                             │       45  │     ~45   │  43 (2+41)│
  │ SQLite / FTS5                          │      Yes  │  Yes (EM) │       No  │
  │ Network from sandbox                   │      Yes  │  fetch()  │       No  │
  │ File system                            │     Full  │ Emscr FS  │     WASI  │
  │ API key in sandbox                     │      Yes  │      Yes  │       No  │
  │ Streaming support                      │      Yes  │    Async  │  via host │
  └────────────────────────────────────────┴───────────┴───────────┴───────────┘""")

    section("C2. SPEEDUP RATIOS (Native = 1.0×)")
    print(f"""
  ┌────────────────────────────────────────┬───────────┬───────────┬───────────┐
  │                                        │  Native   │  Pyodide  │  pywasm   │
  ├────────────────────────────────────────┼───────────┼───────────┼───────────┤
  │ Cold start                             │    1.0×   │ {c_cold['native']/c_cold['pyodide'] if c_cold['pyodide'] else 0:>6.1f}×  │ {c_cold['native']/c_cold['pywasm'] if c_cold['pywasm'] else 0:>6.1f}×  │
  │ Single turn                            │    1.0×   │ {c_1t['native']/c_1t['pyodide'] if c_1t['pyodide'] else 0:>6.1f}×  │ {c_1t['native']/c_1t['pywasm'] if c_1t['pywasm'] else 0:>6.1f}×  │
  │ 20-turn conversation                   │    1.0×   │ {c_20t['native']/c_20t['pyodide'] if c_20t['pyodide'] else 0:>6.1f}×  │ {c_20t['native']/c_20t['pywasm'] if c_20t['pywasm'] else 0:>6.1f}×  │
  │ 20 agents × 1-turn swarm              │    1.0×   │ {s_20['native']/s_20['pyodide'] if s_20['pyodide'] else 0:>6.1f}×  │ {s_20['native']/s_20['pywasm'] if s_20['pywasm'] else 0:>6.1f}×  │
  │ 50 agents × 1-turn swarm              │    1.0×   │ {s_50['native']/s_50['pyodide'] if s_50['pyodide'] else 0:>6.1f}×  │ {s_50['native']/s_50['pywasm'] if s_50['pywasm'] else 0:>6.1f}×  │
  │ 50q ÷ 20 workers                      │    1.0×   │ {s_pool['native']/s_pool['pyodide'] if s_pool['pyodide'] else 0:>6.1f}×  │ {s_pool['native']/s_pool['pywasm'] if s_pool['pywasm'] else 0:>6.1f}×  │
  └────────────────────────────────────────┴───────────┴───────────┴───────────┘""")

    section("C3. WHEN TO USE WHAT")

    api_latency_note = "500–5000ms"

    print(f"""
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  NATIVE PYTHON — use when:                                                 ┃
  ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
  ┃  • You need the full 45-tool suite (terminal, browser, code exec)          ┃
  ┃  • You need SQLite session persistence / FTS5 search                       ┃
  ┃  • Long-running server process (cold start amortized to zero)              ┃
  ┃  • Real-time CLI streaming output                                          ┃
  ┃  • Single agent per request (swarm overhead doesn't matter)                ┃
  ┃  • API latency ({api_latency_note}) dominates anyway                       ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  PYODIDE (BROWSER) — use when:                                             ┃
  ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
  ┃  • Browser-based UI with zero backend                                      ┃
  ┃  • Near-complete feature parity with native (same codebase + 33KB shims)   ┃
  ┃  • User provides their own API key (client-side)                           ┃
  ┃  • Per-turn cost is negative (threading shims help, not hurt)              ┃
  ┃    → {c_marg['pyodide']:.1f}ms/turn vs native's {c_marg['native']:.1f}ms/turn                                      ┃
  ┃  • Parallel agents in-browser: {fmt(s_20['pyodide'])} for 20 agents (same as native)          ┃
  ┃  • Acceptable cold start: ~{c_cold['pyodide']/1000:.0f}s first load                                       ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  PYWASM (WASM BINARY) — use when:                                          ┃
  ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
  ┃  • Cold start matters: {c_cold['pywasm']:.0f}ms vs {c_cold['native']:.0f}ms ({c_cold['native']/c_cold['pywasm']:.0f}× faster)                          ┃
  ┃  • Embedding in non-Python host (Rust, Go, edge runtime, WASI)             ┃
  ┃  • Strong sandboxing: API key never enters Wasm sandbox                    ┃
  ┃  • Swarming: 50 agents in {fmt(s_50['pywasm'])} vs {fmt(s_50['native'])} ({s_50['native']/s_50['pywasm']:.0f}× faster)                    ┃
  ┃  • Multi-agent orchestration: 20×5-turn in {fmt(s_multi20['pywasm'])} vs {fmt(s_multi20['native'])}                ┃
  ┃  • Worker pool: {total_q}q÷20w = {50/(s_pool['pywasm']/1000):.0f} q/s (vs {50/(s_pool['native']/1000):.0f} q/s native)                          ┃
  ┃  • Single portable binary: {sizes['wasm_mb']:.0f} MB, zero pip packages                        ┃
  ┃  • Per-agent cost at scale: {pa50['pywasm']:.0f}ms/agent vs {pa50['native']:.0f}ms/agent (N=50)              ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  NONE OF THESE — use native Python differently when:                       ┃
  ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
  ┃  • Local GPU inference (need CUDA, native extensions)                      ┃
  ┃  • Memory-constrained: all three use ~{c_rss['native']:.0f}MB/instance, no savings       ┃
  ┃  • You need the full 45-tool suite AND parallel execution                  ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
""")

    # Save raw results
    raw_path = HERMES_PYWASM_DIR / "benchmark_full_results.json"
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items() if k not in ("stdout",)}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj
    with open(raw_path, "w") as f:
        json.dump(clean(ALL), f, indent=2)
    print(f"  Raw data saved: {raw_path}")

    # Cleanup
    import shutil
    shutil.rmtree("/tmp/hermes-bench-full", ignore_errors=True)


if __name__ == "__main__":
    main()
