#!/bin/bash
set -euo pipefail

# Build hermes_agent.wasm from Python source using py2wasm.
#
# Prerequisites:
#   pip install py2wasm   (needs Python 3.11)
#   brew install wasmtime  (for testing)
#
# Usage:
#   ./build.sh            # compile only
#   ./build.sh test       # compile + run tests
#   ./build.sh clean      # remove build artifacts

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$SCRIPT_DIR/hermes_agent.wasm"

case "${1:-build}" in
    build)
        echo "Building hermes_agent.wasm..."
        py2wasm "$SCRIPT_DIR/wasm_agent.py" -o "$OUTPUT"
        SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
        echo "✅ Built: $OUTPUT ($SIZE)"
        ;;

    test)
        echo "Running Python protocol tests..."
        python3 "$SCRIPT_DIR/tests/test_protocol.py"

        if [ -f "$OUTPUT" ]; then
            echo ""
            echo "Running Wasm smoke test..."
            RESULT=$(printf '%s\n' \
                '{"type":"init","config":{"model":"test","max_iterations":3}}' \
                '{"type":"user_message","content":"ping"}' \
                '{"type":"memory_data","memory":"","user":""}' \
                '{"type":"skills_data","skills":[]}' \
                '{"type":"api_response","response":{"id":"c","object":"chat.completion","created":0,"model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}}' \
                | timeout 15 wasmtime "$OUTPUT" 2>/dev/null | tail -1)

            if echo "$RESULT" | grep -q '"done"'; then
                echo "✅ Wasm smoke test passed"
            else
                echo "❌ Wasm smoke test failed: $RESULT"
                exit 1
            fi
        else
            echo "⚠  No .wasm file found — run './build.sh' first to test Wasm"
        fi
        ;;

    clean)
        rm -f "$OUTPUT"
        echo "Cleaned."
        ;;

    *)
        echo "Usage: $0 [build|test|clean]"
        exit 1
        ;;
esac
