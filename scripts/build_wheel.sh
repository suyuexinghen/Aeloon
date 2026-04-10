#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/dist"
PYTHON_BIN="${PYTHON:-python3}"
CLEAR_OUTPUT=false

usage() {
    cat <<EOF
Build an Aeloon wheel package.

Usage:
  bash scripts/build_wheel.sh [output-dir] [--clear] [--python /path/to/python]

Examples:
  bash scripts/build_wheel.sh
  bash scripts/build_wheel.sh ./dist
  bash scripts/build_wheel.sh ./artifacts --clear
  bash scripts/build_wheel.sh --python python3.12

Environment:
  PYTHON  Preferred Python interpreter for pip wheel builds (default: python3)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --clear)
            CLEAR_OUTPUT=true
            shift
            ;;
        --python)
            [ $# -ge 2 ] || {
                echo "--python requires a value." >&2
                exit 1
            }
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [ "$OUTPUT_DIR" = "$ROOT_DIR/dist" ]; then
                OUTPUT_DIR="$1"
                shift
            else
                echo "Unknown argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

if [ "$CLEAR_OUTPUT" = true ]; then
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip wheel \
        "$ROOT_DIR" \
        --no-deps \
        --wheel-dir "$OUTPUT_DIR"
elif command -v uv >/dev/null 2>&1; then
    uv build \
        "$ROOT_DIR" \
        --wheel \
        --out-dir "$OUTPUT_DIR" \
        --no-build-logs
else
    echo "Unable to build the wheel: neither '$PYTHON_BIN -m pip' nor 'uv' is available." >&2
    exit 1
fi

WHEEL_FILE="$(
    find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.whl' -print | sort | tail -n 1
)"

if [ -z "$WHEEL_FILE" ]; then
    echo "Wheel build finished, but no .whl file was found in $OUTPUT_DIR." >&2
    exit 1
fi

printf 'Wheel created at %s\n' "$WHEEL_FILE"
