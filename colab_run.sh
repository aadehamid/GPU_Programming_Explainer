#!/usr/bin/env zsh
# Usage: ./colab_run.sh <path-to-file.cu>
# Reads the Colab session ID from .vscode/settings.json, uploads the .cu file,
# compiles it on the T4, and runs it.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
SETTINGS="$SCRIPT_DIR/.vscode/settings.json"
CU_FILE="${1:?Usage: colab_run.sh <path/to/file.cu>}"
BASENAME="${CU_FILE:t}"          # filename only, e.g. hello.cu
STEM="${BASENAME:r}"             # without extension, e.g. hello

# Guard: only accept .cu files
if [[ "${CU_FILE:e}" != "cu" ]]; then
  echo "[colab] ERROR: '${BASENAME}' is not a .cu file. Open a .cu file in the editor first."
  exit 1
fi

# Extract session ID from settings.json
# Uses grep to avoid JSONC comment issues with the standard json module
SESSION=$(grep '"colab.sessionId"' "$SETTINGS" | sed 's/.*"colab.sessionId"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

if [[ -z "$SESSION" ]]; then
  echo "[colab] ERROR: colab.sessionId not set in .vscode/settings.json"
  exit 1
fi

echo "[colab] Session: $SESSION"

if ! colab status -s "$SESSION" >/dev/null; then
  echo "[colab] ERROR: session '$SESSION' is not active."
  echo "[colab] Start a new session with: colab new --gpu T4"
  echo "[colab] Then update colab.sessionId in .vscode/settings.json."
  exit 1
fi

echo "[colab] Uploading $BASENAME..."
colab upload -s "$SESSION" "$CU_FILE" "/content/$BASENAME"

echo "[colab] Compiling and running..."
COMMAND=$(printf '!nvcc -arch=sm_75 /content/%s.cu -o /content/%s && /content/%s\n' \
  "$STEM" "$STEM" "$STEM")

for attempt in 1 2; do
  if print -r -- "$COMMAND" | colab exec -s "$SESSION" --timeout 120; then
    exit 0
  fi

  if [[ "$attempt" == 1 ]]; then
    echo "[colab] Execution connection failed; retrying once..."
    sleep 2
  fi
done

echo "[colab] ERROR: execution failed after retry. Check that session '$SESSION' is still connected."
exit 1
