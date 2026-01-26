import os

import modal
from dotenv import load_dotenv

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
    .uv_pip_install("torch", "numpy")
    # Install development tools
    .apt_install(
        "git", "curl", "make", "gcc", "g++", "ripgrep", "fd-find", 
        "build-essential"
    )
    # Install Neovim
    .run_commands(
        "curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz"
    )
    .run_commands("rm -rf /opt/nvim-linux-x86_64")
    .run_commands("mkdir -p /opt/nvim-linux-x86_64")
    .run_commands("tar -C /opt -xzf nvim-linux-x86_64.tar.gz")
    .run_commands("ln -sf /opt/nvim-linux-x86_64/bin/nvim /usr/local/bin/nvim")
    .run_commands("rm nvim-linux-x86_64.tar.gz")
    # Install uv and Mojo
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .run_commands(
        "cd /home && /root/.local/bin/uv venv && . .venv/bin/activate && /root/.local/bin/uv pip install mojo \
                      --index-url https://dl.modular.com/public/nightly/python/simple/ \
                      --prerelease allow"
    )
)

app = modal.App("CUDA-images-minimal")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("github-secret")],
    gpu="T4",  # or "A10G", "A100", etc.
    cpu=8,
    memory=32768,
    timeout=86400,  # 24 hours
)
def dev_shell():
    """Interactive development environment with SSH access"""
    import time
    
    print("Development environment ready!")
    print(f"Working directory: {os.getcwd()}")
    
    # Keep the container running
    while True:
        time.sleep(60)
