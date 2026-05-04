#!/usr/bin/env python3
"""
Comprehensive benchmark: Native Python vs Pyodide (simulated) vs pywasm (Wasm)

Measures:
  - Wall-clock time (perf_counter)
  - Peak memory RSS (via /usr/bin/time -l on macOS)
  - CPU time (user + sys)
  - Binary / deployment size
  - Startup vs per-turn amortization
  - Scaling behavior (1, 5, 10, 20 turns)
  - Concurrent instance overhead
  - Sandboxing surface area (qualitative)
"""

import json
import math
import os
import re
import subprocess
import sys
import time
import statistics
import textwrap
from pathlib import Path

HERMES_AGENT_DIR = Path(__file__).parent.parent / "hermes-agent"
HERMES_PYODIDE_DIR = Path(__file__).parent.parent / "hermes-pyodide"
HERMES_PYWASM_DIR = Path(__file__).parent
WASM_BINARY = HERMES_PYWASM_DIR / "hermes_agent.wasm"

NATIVE_PY = str(HERMES_AGENT_DIR / "venv/bin/python3")
RUNS = 5

# ── Mock data ──

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
# Measurement helpers
# ═══════════════════════════════════════════════════════════════════════════

def measure_process(cmd, stdin_data=None, timeout=120):
    """Run a command via /usr/bin/time -l, return detailed metrics.

    Returns dict with: wall_ms, user_ms, sys_ms, rss_kb, page_faults,
                       ctx_voluntary, ctx_involuntary, stdout, stderr
    """
    full_cmd = ["/usr/bin/time", "-l"] + cmd
    proc = subprocess.run(
        full_cmd,
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    stderr = proc.stderr or ""

    def extract(pattern, default=0):
        m = re.search(pattern, stderr)
        return float(m.group(1)) if m else default

    wall = extract(r'([\d.]+)\s+real')
    user = extract(r'([\d.]+)\s+user')
    sys_ = extract(r'([\d.]+)\s+sys')
    rss_bytes = extract(r'(\d+)\s+maximum resident set size')
    page_faults = extract(r'(\d+)\s+page faults')
    ctx_vol = extract(r'(\d+)\s+voluntary context switches')
    ctx_inv = extract(r'(\d+)\s+involuntary context switches')

    return {
        "wall_ms": wall * 1000,
        "user_ms": user * 1000,
        "sys_ms": sys_ * 1000,
        "cpu_ms": (user + sys_) * 1000,
        "rss_kb": rss_bytes / 1024,
        "rss_mb": rss_bytes / (1024 * 1024),
        "page_faults": int(page_faults),
        "ctx_voluntary": int(ctx_vol),
        "ctx_involuntary": int(ctx_inv),
        "stdout": proc.stdout or "",
        "ok": proc.returncode == 0,
    }


def make_native_script(bench_code):
    """Wrap bench code into a full native-hermes script."""
    return NATIVE_PREAMBLE + MOCK_SETUP + AGENT_INIT + bench_code


def make_pyodide_script(bench_code):
    """Wrap bench code into a full pyodide-simulated script."""
    return PYODIDE_PREAMBLE + MOCK_SETUP + AGENT_INIT + bench_code


def make_wasm_input(messages_and_responses):
    """Build JSON-lines input for the wasm agent."""
    lines = [{"type": "init", "config": {"model": "bench", "max_iterations": 50}}]
    for user_msg, *responses in messages_and_responses:
        lines.append({"type": "user_message", "content": user_msg})
        lines.append({"type": "memory_data", "memory": "", "user": ""})
        lines.append({"type": "skills_data", "skills": []})
        for r in responses:
            lines.append({"type": "api_response", "response": r})
    return "\n".join(json.dumps(m) for m in lines) + "\n"


# ── Script preambles ──

NATIVE_PREAMBLE = textwrap.dedent(f"""\
import time, os, sys, json
os.environ['HERMES_HOME'] = '/tmp/hermes-bench'
os.makedirs('/tmp/hermes-bench/logs', exist_ok=True)
sys.path.insert(0, {str(HERMES_AGENT_DIR)!r})
import httpx
""")

PYODIDE_PREAMBLE = textwrap.dedent(f"""\
import time, os, sys, json
os.environ['HERMES_HOME'] = '/tmp/hermes-bench'
os.makedirs('/tmp/hermes-bench/logs', exist_ok=True)
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


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark functions
# ═══════════════════════════════════════════════════════════════════════════

def bench_init_only():
    """Import + AIAgent initialization, no conversation."""
    results = {"native": [], "pyodide": [], "pywasm": []}

    init_script = "pass  # agent already initialized above\n"
    native_s = make_native_script(init_script)
    pyodide_s = make_pyodide_script(init_script)
    wasm_in = json.dumps({"type": "init", "config": {"model": "bench"}}) + "\n"

    for _ in range(RUNS):
        results["native"].append(measure_process([NATIVE_PY, "-c", native_s]))
        results["pyodide"].append(measure_process([NATIVE_PY, "-c", pyodide_s]))
        results["pywasm"].append(measure_process(
            ["wasmtime", str(WASM_BINARY)], stdin_data=wasm_in))
    return results


def bench_single_turn():
    """One user message → one API response."""
    results = {"native": [], "pyodide": [], "pywasm": []}

    conv = textwrap.dedent("""\
queue[:] = [SIMPLE]; idx[0] = 0
agent.run_conversation(user_message='Hello')
""")
    native_s = make_native_script(conv)
    pyodide_s = make_pyodide_script(conv)
    wasm_in = make_wasm_input([("Hello", SIMPLE)])

    for _ in range(RUNS):
        results["native"].append(measure_process([NATIVE_PY, "-c", native_s]))
        results["pyodide"].append(measure_process([NATIVE_PY, "-c", pyodide_s]))
        results["pywasm"].append(measure_process(
            ["wasmtime", str(WASM_BINARY)], stdin_data=wasm_in))
    return results


def bench_tool_call():
    """Memory tool call → follow-up → done (2 API calls)."""
    results = {"native": [], "pyodide": [], "pywasm": []}

    conv = textwrap.dedent("""\
queue[:] = [MEM_TOOL, FINAL]; idx[0] = 0
agent.run_conversation(user_message='Remember I like pizza')
""")
    native_s = make_native_script(conv)
    pyodide_s = make_pyodide_script(conv)
    wasm_in = make_wasm_input([("Remember I like pizza", MEM_TOOL, FINAL)])

    for _ in range(RUNS):
        results["native"].append(measure_process([NATIVE_PY, "-c", native_s]))
        results["pyodide"].append(measure_process([NATIVE_PY, "-c", pyodide_s]))
        results["pywasm"].append(measure_process(
            ["wasmtime", str(WASM_BINARY)], stdin_data=wasm_in))
    return results


def bench_scaling(turn_counts=(1, 5, 10, 20)):
    """Measure how time/memory scale with conversation length."""
    results = {}
    for n in turn_counts:
        results[n] = {"native": [], "pyodide": [], "pywasm": []}

        msgs = [f"Message {i+1}" for i in range(n)]
        conv = "history = []\n"
        for m in msgs:
            conv += f"queue[:] = [SIMPLE]; idx[0] = 0\n"
            conv += f"r = agent.run_conversation(user_message={m!r}, conversation_history=history)\n"
            conv += "history = r['messages']\n"

        native_s = make_native_script(conv)
        pyodide_s = make_pyodide_script(conv)
        wasm_in = make_wasm_input([(m, SIMPLE) for m in msgs])

        for _ in range(RUNS):
            results[n]["native"].append(measure_process([NATIVE_PY, "-c", native_s]))
            results[n]["pyodide"].append(measure_process([NATIVE_PY, "-c", pyodide_s]))
            results[n]["pywasm"].append(measure_process(
                ["wasmtime", str(WASM_BINARY)], stdin_data=wasm_in))
    return results


def bench_concurrent(instance_counts=(1, 3, 5)):
    """Launch N agent instances simultaneously, measure total resources."""
    results = {}

    conv = textwrap.dedent("""\
queue[:] = [SIMPLE]; idx[0] = 0
agent.run_conversation(user_message='Hello')
""")
    native_s = make_native_script(conv)
    wasm_in = make_wasm_input([("Hello", SIMPLE)])

    for n in instance_counts:
        results[n] = {"native": [], "pywasm": []}
        for _ in range(RUNS):
            # Native: launch N python subprocesses in parallel
            t0 = time.perf_counter()
            procs_n = [subprocess.Popen([NATIVE_PY, "-c", native_s],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(n)]
            for p in procs_n: p.wait()
            wall_n = (time.perf_counter() - t0) * 1000
            results[n]["native"].append({"wall_ms": wall_n, "instances": n})

            # Wasm: launch N wasmtime processes in parallel
            t0 = time.perf_counter()
            procs_w = [subprocess.Popen(["wasmtime", str(WASM_BINARY)],
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True) for _ in range(n)]
            for p in procs_w:
                p.communicate(input=wasm_in)
            wall_w = (time.perf_counter() - t0) * 1000
            results[n]["pywasm"].append({"wall_ms": wall_w, "instances": n})

    return results


def measure_sizes():
    """Deployment artifact sizes."""
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

    source_size = dir_size(HERMES_AGENT_DIR,
                           exclude={'.git', 'venv', '__pycache__', 'node_modules', '.vers'})
    venv_size = dir_size(HERMES_AGENT_DIR / "venv/lib")
    wasm_size = WASM_BINARY.stat().st_size if WASM_BINARY.exists() else 0
    pyodide_patch_size = sum(
        f.stat().st_size for f in HERMES_PYODIDE_DIR.rglob("*.py")
        if "hermes-agent" not in str(f) and "__pycache__" not in str(f))

    pkg_count = len(list(
        (HERMES_AGENT_DIR / "venv/lib/python3.11/site-packages").glob("*.dist-info")))

    return {
        "native_source_mb": source_size / (1024**2),
        "native_venv_mb": venv_size / (1024**2),
        "native_total_mb": (source_size + venv_size) / (1024**2),
        "native_pip_packages": pkg_count,
        "pyodide_patch_kb": pyodide_patch_size / 1024,
        "pyodide_total_mb": (source_size + venv_size) / (1024**2),  # same as native + patches
        "pywasm_binary_mb": wasm_size / (1024**2),
        "pywasm_host_runner_kb": (HERMES_PYWASM_DIR / "host_runner.py").stat().st_size / 1024,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════════

def med(values):
    return statistics.median(values) if values else 0

def sd(values):
    return statistics.stdev(values) if len(values) >= 2 else 0

def extract_metric(runs, key):
    return [r[key] for r in runs if key in r]


def print_section(title):
    print(f"\n{'━' * 78}")
    print(f"  {title}")
    print(f"{'━' * 78}")


def print_metric_table(results, metrics, targets=("native", "pyodide", "pywasm")):
    labels = {"native": "Native Python", "pyodide": "Pyodide (sim)",
              "pywasm": "pywasm (Wasm)"}
    col_w = 18

    header = f"  {'':20}"
    for t in targets:
        header += f"{labels[t]:>{col_w}}"
    print(header)
    print(f"  {'─'*20}" + f"{'─'*col_w}" * len(targets))

    for metric_key, metric_label, fmt, unit in metrics:
        row = f"  {metric_label:20}"
        for t in targets:
            vals = extract_metric(results.get(t, []), metric_key)
            if vals:
                m = med(vals)
                s = sd(vals)
                row += f"{f'{m:{fmt}}{unit} ±{s:.0f}':>{col_w}}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)


def main():
    if not WASM_BINARY.exists():
        print("ERROR: Build hermes_agent.wasm first: ./build.sh")
        sys.exit(1)

    print("=" * 78)
    print("  HERMES AGENT — COMPREHENSIVE BENCHMARK")
    print(f"  Native Python vs Pyodide (simulated) vs pywasm (compiled Wasm)")
    print(f"  {RUNS} runs per test, median ± stdev reported")
    print("=" * 78)

    # ── 1. Deployment size ──
    print_section("1. DEPLOYMENT SIZE")
    sizes = measure_sizes()
    print(f"""
  Native Python (stock hermes-agent):
    Source code:          {sizes['native_source_mb']:>8.1f} MB   ({sizes['native_pip_packages']} pip packages)
    venv (site-packages): {sizes['native_venv_mb']:>8.1f} MB
    Total on disk:        {sizes['native_total_mb']:>8.1f} MB
    Runtime requires:     Python 3.11 + pip install

  Pyodide (browser):
    Same as native +      {sizes['pyodide_patch_kb']:>8.1f} KB of patches/shims
    Browser download:     ~20 MB (Pyodide core) + ~5 MB (packages via micropip)
    Runtime requires:     Modern browser with WebAssembly

  pywasm (compiled Wasm):
    Single binary:        {sizes['pywasm_binary_mb']:>8.1f} MB   (hermes_agent.wasm)
    Host runner:          {sizes['pywasm_host_runner_kb']:>8.1f} KB   (host_runner.py, or any language)
    Runtime requires:     wasmtime/wasmer (no Python needed)
""")

    # ── 2. Cold start ──
    print_section("2. COLD START (import + init, no conversation)")
    r_init = bench_init_only()
    print_metric_table(r_init, [
        ("wall_ms",  "Wall clock",      ".0f", "ms"),
        ("cpu_ms",   "CPU time",        ".0f", "ms"),
        ("rss_mb",   "Peak RSS",        ".1f", "MB"),
        ("page_faults", "Page faults",  ".0f", ""),
        ("ctx_voluntary", "Vol ctx sw",  ".0f", ""),
    ])

    # ── 3. Single turn ──
    print_section("3. SINGLE-TURN CONVERSATION (1 user msg → 1 API response)")
    r_single = bench_single_turn()
    print_metric_table(r_single, [
        ("wall_ms",  "Wall clock",      ".0f", "ms"),
        ("cpu_ms",   "CPU time",        ".0f", "ms"),
        ("rss_mb",   "Peak RSS",        ".1f", "MB"),
    ])

    # ── 4. Tool call ──
    print_section("4. TOOL CALL FLOW (memory add → follow-up, 2 API calls)")
    r_tool = bench_tool_call()
    print_metric_table(r_tool, [
        ("wall_ms",  "Wall clock",      ".0f", "ms"),
        ("cpu_ms",   "CPU time",        ".0f", "ms"),
        ("rss_mb",   "Peak RSS",        ".1f", "MB"),
    ])

    # ── 5. Scaling ──
    print_section("5. SCALING — Wall time & memory vs conversation length")
    r_scaling = bench_scaling((1, 5, 10, 20))

    targets = ("native", "pyodide", "pywasm")
    labels = {"native": "Native", "pyodide": "Pyodide", "pywasm": "pywasm"}
    print(f"\n  Wall clock (ms):")
    print(f"  {'Turns':>8}" + "".join(f"{labels[t]:>14}" for t in targets))
    print(f"  {'─'*8}" + "─"*14*len(targets))
    for n in (1, 5, 10, 20):
        row = f"  {n:>8}"
        for t in targets:
            vals = extract_metric(r_scaling[n][t], "wall_ms")
            row += f"{med(vals):>12.0f}ms" if vals else f"{'—':>14}"
        print(row)

    print(f"\n  Peak RSS (MB):")
    print(f"  {'Turns':>8}" + "".join(f"{labels[t]:>14}" for t in targets))
    print(f"  {'─'*8}" + "─"*14*len(targets))
    for n in (1, 5, 10, 20):
        row = f"  {n:>8}"
        for t in targets:
            vals = extract_metric(r_scaling[n][t], "rss_mb")
            row += f"{med(vals):>12.1f}MB" if vals else f"{'—':>14}"
        print(row)

    print(f"\n  Per-turn marginal cost (ms, from 1→20 turns):")
    for t in targets:
        w1 = med(extract_metric(r_scaling[1][t], "wall_ms"))
        w20 = med(extract_metric(r_scaling[20][t], "wall_ms"))
        if w1 and w20:
            marginal = (w20 - w1) / 19
            print(f"  {labels[t]:>12}: {marginal:>6.1f}ms/turn  (total {w1:.0f}ms → {w20:.0f}ms)")

    # ── 6. Concurrent instances ──
    print_section("6. CONCURRENT INSTANCES (wall time for N simultaneous agents)")
    r_conc = bench_concurrent((1, 3, 5))

    print(f"\n  {'N':>4}  {'Native (ms)':>14}  {'pywasm (ms)':>14}  {'Ratio':>8}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*8}")
    for n in (1, 3, 5):
        nv = [r["wall_ms"] for r in r_conc[n]["native"]]
        wv = [r["wall_ms"] for r in r_conc[n]["pywasm"]]
        nm, wm = med(nv), med(wv)
        ratio = nm / wm if wm > 0 else 0
        print(f"  {n:>4}  {nm:>12.0f}ms  {wm:>12.0f}ms  {ratio:>7.1f}×")

    # ── 7. Summary ──
    print_section("7. SUMMARY — WHEN TO USE WHAT")
    # Extract key numbers
    native_cold = med(extract_metric(r_init["native"], "wall_ms"))
    pyodide_cold = med(extract_metric(r_init["pyodide"], "wall_ms"))
    pywasm_cold = med(extract_metric(r_init["pywasm"], "wall_ms"))

    native_1t = med(extract_metric(r_single["native"], "wall_ms"))
    pyodide_1t = med(extract_metric(r_single["pyodide"], "wall_ms"))
    pywasm_1t = med(extract_metric(r_single["pywasm"], "wall_ms"))

    native_rss = med(extract_metric(r_single["native"], "rss_mb"))
    pyodide_rss = med(extract_metric(r_single["pyodide"], "rss_mb"))
    pywasm_rss = med(extract_metric(r_single["pywasm"], "rss_mb"))

    native_5t = med(extract_metric(r_scaling[5]["native"], "wall_ms"))
    pywasm_5t = med(extract_metric(r_scaling[5]["pywasm"], "wall_ms"))

    native_20t = med(extract_metric(r_scaling[20]["native"], "wall_ms"))
    pywasm_20t = med(extract_metric(r_scaling[20]["pywasm"], "wall_ms"))

    conc5_native = med([r["wall_ms"] for r in r_conc[5]["native"]])
    conc5_pywasm = med([r["wall_ms"] for r in r_conc[5]["pywasm"]])

    print(f"""
  ┌───────────────────────────────┬──────────┬──────────┬──────────┐
  │ Metric                        │  Native  │  Pyodide │  pywasm  │
  ├───────────────────────────────┼──────────┼──────────┼──────────┤
  │ Cold start                    │ {native_cold:>6.0f}ms │ {pyodide_cold:>6.0f}ms │ {pywasm_cold:>6.0f}ms │
  │ Single turn (1 API call)      │ {native_1t:>6.0f}ms │ {pyodide_1t:>6.0f}ms │ {pywasm_1t:>6.0f}ms │
  │ 5-turn conversation           │ {native_5t:>6.0f}ms │          │ {pywasm_5t:>6.0f}ms │
  │ 20-turn conversation          │ {native_20t:>6.0f}ms │          │ {pywasm_20t:>6.0f}ms │
  │ Peak memory (single turn)     │ {native_rss:>5.0f}MB │ {pyodide_rss:>5.0f}MB │ {pywasm_rss:>5.0f}MB │
  │ Deployment size               │ {sizes['native_total_mb']:>5.0f}MB │  ~25 MB  │ {sizes['pywasm_binary_mb']:>5.0f}MB │
  │ 5 concurrent (wall)           │ {conc5_native:>6.0f}ms │          │ {conc5_pywasm:>6.0f}ms │
  │ Runtime dependency            │ Py 3.11  │ Browser  │ wasmtime │
  │ Pip packages required         │ {sizes['native_pip_packages']:>6}   │  ~10     │      0   │
  │ Network from sandbox          │   Yes    │  fetch() │    No    │
  │ File system access            │   Full   │ Emscr FS │   WASI   │
  │ API key in sandbox            │   Yes    │   Yes    │    No    │
  └───────────────────────────────┴──────────┴──────────┴──────────┘""")

    print(f"""
  USE NATIVE PYTHON when:
  • You need the full 45-tool suite (terminal, browser, code execution)
  • You need SQLite session persistence / FTS5 search
  • You're running on a server with Python already installed
  • Startup time doesn't matter (long-running process)

  USE PYODIDE when:
  • You need a browser-based UI with no backend server
  • You want near-complete feature parity with native
  • Users provide their own API keys (client-side)
  • ~{pyodide_cold/1000:.0f}s cold start is acceptable

  USE PYWASM when:
  • You're embedding the agent in a non-Python host (Rust, Go, edge runtime)
  • Cold start matters: {pywasm_cold:.0f}ms vs {native_cold:.0f}ms ({native_cold/pywasm_cold:.1f}× faster)
  • You want strong sandboxing (API key never enters the Wasm sandbox)
  • You're running many concurrent instances: 5× agents in {conc5_pywasm:.0f}ms vs {conc5_native:.0f}ms
  • You need a single portable binary (26 MB, no pip install)
  • Per-turn overhead is negligible vs API latency (500-5000ms)

  NONE OF THESE when:
  • You need local GPU inference (use native Python)
  • You need real-time streaming to CLI (use native Python)
""")

    # Save raw data
    raw = {
        "sizes": sizes,
        "init": {t: [r for r in runs] for t, runs in r_init.items()},
        "single_turn": {t: [r for r in runs] for t, runs in r_single.items()},
        "tool_call": {t: [r for r in runs] for t, runs in r_tool.items()},
        "scaling": {str(n): {t: [r for r in runs] for t, runs in v.items()}
                    for n, v in r_scaling.items()},
        "concurrent": {str(n): {t: [r for r in runs] for t, runs in v.items()}
                       for n, v in r_conc.items()},
    }
    raw_path = HERMES_PYWASM_DIR / "benchmark_results.json"
    # Filter out non-serializable fields
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k != "stdout"}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj
    with open(raw_path, "w") as f:
        json.dump(clean(raw), f, indent=2)
    print(f"  Raw data: {raw_path}")


if __name__ == "__main__":
    main()
