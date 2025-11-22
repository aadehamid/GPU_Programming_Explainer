import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import modal
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return modal, os


@app.cell
def _(os):
    GITHUB_TOKEN=os.getenv("GITHUB_TOKEN")
    GITHUB_TOKEN
    return


app._unparsable_cell(
    r"""
    git remote set-url origin https://$GITHUB_TOKEN@github.com/aadehamid/GPU_Programming_Explainer.git
    !git config --global user.name \"aadehamid\"
    !git config --global user.email \"aadehamid@gmail.com\"

    if [ ! -d \"GPU_Programming_Explainer\" ]; then git clone https://github.com/aadehamid/GPU_Programming_Explainer.git; fi
    %cd GPU_Programming_Explainer

    """,
    name="_"
)


@app.cell
def _(modal):
    # Image definitions

    # cuda_version = "12.8.1"  # should be no greater than host CUDA version
    cuda_version = "13.0.2"  # should be no greater than host CUDA version
    # flavor = "devel"  # includes full CUDA toolkit
    flavor = "cudnn-devel"  # includes full CUDA toolkit
    operating_sys = "ubuntu24.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"
    # HF_CACHE_PATH = "/cache"


    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
        .entrypoint([])  # remove verbose logging by base image on entry
        .apt_install("git", "curl")
        .run_commands(
            "git clone https://github.com/aadehamid/GPU_Programming_Explainer.git"
        )
        .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
        .run_commands(
            "uv init && uv venv && source .venv/bin/activate && uv pip install mojo \
                          --index-url https://dl.modular.com/public/nightly/python/simple/ \
                          --prerelease allow")
        .uv_pip_install("torch", "duckdb", "pandas", "rich", "marimo", "numpy")
        .run_commands("rm -rf /root/GPU_Programming_Explainer")
        .run_commands("if [ ! -d 'GPU_Programming_Explainer' ]; then git clone https://github.com/aadehamid/GPU_Programming_Explainer.git; fi")
        .run_commands("cd /root/GPU_Programming_Explainer")
        .run_commands("git remote set-url origin https://$GITHUB_TOKEN@github.com/aadehamid/GPU_Programming_Explainer.git")
        .run_commands("git config --global user.name 'aadehamid'")
        .run_commands('git config --global user.email "aadehamid@gmail.com"')
        # .apt_install("libopenmpi-dev")  # required for tensorrt
        # .pip_install("tensorrt-llm==0.19.0", "pynvml", extra_index_url="https://pypi.nvidia.com")
        # .pip_install("hf-transfer", "huggingface_hub[hf_xet]")
        # .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1", "PMIX_MCA_gds": "hash"})
    )


    app = modal.App("cuda-image", image=image)


    @app.function(image=image)
    def cuda_image():
        pass
    return


if __name__ == "__main__":
    app.run()
