import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import modal
    return (modal,)


@app.cell
def _(HF_CACHE_PATH, modal):
    # Image definitions

    cuda_version = "12.8.1"  # should be no greater than host CUDA version
    flavor = "devel"  # includes full CUDA toolkit
    operating_sys = "ubuntu24.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"
    # HF_CACHE_PATH = "/cache"


    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
        .entrypoint([])  # remove verbose logging by base image on entry
        .apt_install("git")
        .run_commands()
        .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
        .run_commands("uv init && uv venv && source .venv/bin/activate && uv pip install mojo \
                          --index-url https://dl.modular.com/public/nightly/python/simple/ \
                          --prerelease allow")
        # .apt_install("libopenmpi-dev")  # required for tensorrt
        # .pip_install("tensorrt-llm==0.19.0", "pynvml", extra_index_url="https://pypi.nvidia.com")
        # .pip_install("hf-transfer", "huggingface_hub[hf_xet]")
        # .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1", "PMIX_MCA_gds": "hash"})
    
    )


    app = modal.App("tensorrt-llm", image=image)
    hf_cache_volume = modal.Volume.from_name("hf_cache_tensorrt", create_if_missing=True)


    @app.function(gpu="A10G", volumes={HF_CACHE_PATH: hf_cache_volume})
    def run_tiny_model():
        from tensorrt_llm import LLM, SamplingParams

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        output = llm.generate("The capital of France is", sampling_params)
        print(f"Generated text: {output.outputs[0].text}")
        return output.outputs[0].text
    return


if __name__ == "__main__":
    app.run()
