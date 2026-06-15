# CUDA VS Code + Colab Setup

This guide sets up a Mac folder so VS Code/NVIDIA Nsight can edit CUDA files locally while Google Colab compiles and runs them on an NVIDIA GPU.

## Quick Copy List For A New Folder

To reuse this setup in a new CUDA workspace, copy these items:

```text
.vscode/
colab_run.sh
CUDA_VSCODE_COLAB_SETUP.md
.cuda/
```

After copying, make sure the run script is executable:

```bash
chmod +x colab_run.sh
```

If you start a new Colab session, update this value in `.vscode/settings.json`:

```json
"colab.sessionId": "YOUR_NEW_SESSION_ID"
```

Keep `.cuda/include/`. It contains the mirrored CUDA headers used by local IntelliSense.

## Goal

- Use VS Code on Mac for `.cu` and `.cuh` editing.
- Use NVIDIA Nsight VS Code Edition for CUDA syntax highlighting.
- Use local copied CUDA headers for C/C++ IntelliSense.
- Use Google Colab for actual `nvcc` compilation and GPU execution.

Modern macOS does not support a current local CUDA toolkit/runtime, so the Mac is used as the editor and Colab is used as the CUDA machine.

## Required VS Code Extensions

Install these extensions:

- `NVIDIA.nsight-vscode-edition`
- `Google.colab`
- `ms-toolsai.jupyter`
- `ms-vscode.cpptools`

You can recommend them in a new workspace with `.vscode/extensions.json`:

```json
{
  "recommendations": [
    "NVIDIA.nsight-vscode-edition",
    "Google.colab",
    "ms-toolsai.jupyter",
    "ms-vscode.cpptools"
  ]
}
```

## Start A Colab GPU Session

From the workspace folder:

```bash
colab new --gpu T4
```

Example output:

```text
[colab] Creating session 'cdf046'...
[colab] Session READY.
```

Save the session ID. In this example it is `cdf046`.

## Copy CUDA Headers From Colab

Create a local folder for mirrored CUDA headers:

```bash
mkdir -p .cuda/include
```

Create a header archive inside Colab:

```bash
printf '!tar -C /usr/local/cuda/include -czf /content/cuda-include.tgz .\n' \
  | colab exec -s YOUR_SESSION_ID --timeout 120
```

Download the archive:

```bash
colab download -s YOUR_SESSION_ID /content/cuda-include.tgz .cuda/cuda-include.tgz
```

Extract it locally:

```bash
tar -xzf .cuda/cuda-include.tgz -C .cuda/include
```

Verify the important headers exist:

```bash
test -f .cuda/include/cuda_runtime.h
test -f .cuda/include/cuda.h
```

## Configure VS Code

Create or update `.vscode/settings.json`:

```json
{
  "files.associations": {
    "*.cu": "cuda-cpp",
    "*.cuh": "cuda-cpp"
  },

  "C_Cpp.errorSquiggles": "enabled",
  "C_Cpp.intelliSenseEngine": "Default",
  "C_Cpp.default.compilerPath": "/usr/bin/clang++",
  "C_Cpp.default.intelliSenseMode": "macos-clang-arm64",
  "C_Cpp.default.cppStandard": "c++17",
  "C_Cpp.default.includePath": [
    "${workspaceFolder}/**",
    "${workspaceFolder}/.cuda/include"
  ],
  "C_Cpp.default.defines": [
    "__CUDACC__"
  ],

  "colab.sessionId": "YOUR_SESSION_ID",

  "editor.formatOnSave": true,
  "editor.tabSize": 2,
  "files.trimTrailingWhitespace": true
}
```

Use `macos-clang-arm64` for Apple Silicon Macs.

Use `macos-clang-x64` for Intel Macs.

After changing settings, run this in VS Code:

```text
Command Palette -> C/C++: Reset IntelliSense Database
```

Then reopen the `.cu` file.

## Optional Build Task

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Colab: Upload & Run current .cu file",
      "type": "shell",
      "command": "${workspaceFolder}/colab_run.sh",
      "args": ["${file}"],
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "presentation": {
        "reveal": "always",
        "panel": "shared",
        "clear": true
      },
      "problemMatcher": []
    }
  ]
}
```

Then `Cmd + Shift + B` runs the current CUDA file through Colab.

## Optional Run Script

Create `colab_run.sh`:

```zsh
#!/usr/bin/env zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
SETTINGS="$SCRIPT_DIR/.vscode/settings.json"
CU_FILE="${1:?Usage: colab_run.sh <path/to/file.cu>}"
BASENAME="${CU_FILE:t}"
STEM="${BASENAME:r}"

if [[ "${CU_FILE:e}" != "cu" ]]; then
  echo "[colab] ERROR: '${BASENAME}' is not a .cu file. Open a .cu file in the editor first."
  exit 1
fi

SESSION=$(grep '"colab.sessionId"' "$SETTINGS" | sed 's/.*"colab.sessionId"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

if [[ -z "$SESSION" ]]; then
  echo "[colab] ERROR: colab.sessionId not set in .vscode/settings.json"
  exit 1
fi

echo "[colab] Session: $SESSION"
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
```

Make it executable:

```bash
chmod +x colab_run.sh
```

## Notes

- Syntax highlighting comes from NVIDIA Nsight and the `.cu`/`.cuh` file associations.
- IntelliSense comes from the Microsoft C/C++ extension and the copied `.cuda/include` headers.
- Compilation and GPU execution still happen on Colab.
- VS Code may still show occasional false positives around CUDA kernel launch syntax such as `kernel<<<blocks, threads>>>()`; the real check is the Colab `nvcc` build.
- If Colab disconnects, start a new session with `colab new --gpu T4`, then update `colab.sessionId`.
