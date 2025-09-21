
__generated_with = "0.16.0"

# %%
# import marimo as mo
import modal

# %%
# Image definitions

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
# HF_CACHE_PATH = "/cache"


image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git", "curl")
    # .run_commands(
    #     "git clone https://github.com/aadehamid/GPU_Programming_Explainer.git"
    # )
    .run_commands(
        "/bin/bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh && "
        "export PATH=$HOME/.local/bin:$PATH && "
        "uv pip install --system mojo "
        "--index-url https://dl.modular.com/public/nightly/python/simple/ "
        "--prerelease allow'"
    )
    # .apt_install("libopenmpi-dev")  # required for tensorrt
    # .pip_install("tensorrt-llm==0.19.0", "pynvml", extra_index_url="https://pypi.nvidia.com")
    # .pip_install("hf-transfer", "huggingface_hub[hf_xet]")
    # .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1", "PMIX_MCA_gds": "hash"})
    .add_local_dir(".", remote_path="/root/GPU_Programming_Explainer", copy=True)
)


app = modal.App("cuda-image", image=image)


@app.function(image=image)
def cuda_image():
    pass
