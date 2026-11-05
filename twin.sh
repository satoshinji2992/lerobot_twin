#!/bin/sh
# Twin Depot launcher for macOS / Linux:  ./twin.sh demo | real | ports | setup | ...  (no argument: web control panel)
here=$(cd "$(dirname "$0")" && pwd)
py=""
[ -f "$here/.twin-python" ] && py=$(head -n 1 "$here/.twin-python")
if [ -z "$py" ] || [ ! -x "$py" ]; then
    py=$(command -v python3 || command -v python)
fi
if [ -z "$py" ]; then
    echo "[twin] Python 3 not found. macOS: install Miniconda or 'brew install python'; Linux: your package manager." >&2
    exit 1
fi
exec "$py" "$here/twin.py" "$@"
