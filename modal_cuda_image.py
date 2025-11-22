import modal
from dotenv import load_dotenv
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
cuda_version = "13.0.2"  # should be no greater than host CUDA version
flavor = "cudnn-devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{tag}",
        add_python="3.12",
        secrets=[modal.Secret.from_name("github-secret")],
    )
    .entrypoint([])
    .pip_install("GitPython")
    .uv_pip_install("torch", "duckdb", "pandas", "rich", "marimo", "numpy")
    .apt_install("git", "curl")
    .run_commands("rm -rf /home/GPU_Programming_Explainer")
    .run_commands(
        "cd /home && git clone https://github.com/aadehamid/GPU_Programming_Explainer.git"
    )
    .run_commands(
        "cd /home/GPU_Programming_Explainer &&  git remote set-url origin https://$GITHUB_TOKEN@github.com/aadehamid/GPU_Programming_Explainer.git"
    )
    .run_commands("git config --global user.name 'aadehamid'")
    .run_commands('git config --global user.email "aadehamid@gmail.com"')
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .run_commands(
        "cd /home && /root/.local/bin/uv venv && . .venv/bin/activate && /root/.local/bin/uv pip install mojo \
                      --index-url https://dl.modular.com/public/nightly/python/simple/ \
                      --prerelease allow"
    )
)

app = modal.App("CUDA-images")


@app.function(
    image=image, secrets=[modal.Secret.from_name("github-secret")]
)  # You need a Function object to reference the image.
def cuda_image():
    pass
