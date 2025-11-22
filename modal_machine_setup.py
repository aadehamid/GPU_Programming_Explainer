__generated_with = "0.16.0"

# %%
# import marimo as mo
import modal
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# %%
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_TOKEN

# %%
# Image definitions

# cuda_version = "12.9.1"  # should be no greater than host CUDA version
cuda_version = "13.0.2"  # should be no greater than host CUDA version
# flavor = "devel"  # includes full CUDA toolkit
flavor = "cudnn-devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
# HF_CACHE_PATH = "/cache"


image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{tag}",
        add_python="3.12",
        secrets=[modal.Secret.from_name("github-secret")],
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .pip_install("GitPython")
    .uv_pip_install("torch", "duckdb", "pandas", "rich", "marimo", "numpy")
    .apt_install("git", "curl")
    .run_commands("cd ~ && rm -rf ./GPU_Programming_Explainer")
    .run_commands(
        "if [ ! -d './GPU_Programming_Explainer' ]; then git clone https://github.com/aadehamid/GPU_Programming_Explainer.git; fi"
    )
    .run_commands(
        "cd ./GPU_Programming_Explainer && git remote set-url origin https://$GITHUB_TOKEN@github.com/aadehamid/GPU_Programming_Explainer.git"
    )
    # .run_commands(
    #     "cd /GPU_Programming_Explainer && git remote set-url origin https://$GITHUB_TOKEN@github.com/aadehamid/GPU_Programming_Explainer.git"
    # )
    # .run_commands(
    #     "git remote set-url origin https://$GITHUB_TOKEN@github.com/aadehamid/GPU_Programming_Explainer.git"
    # )
    .run_commands("git config --global user.name 'aadehamid'")
    .run_commands('git config --global user.email "aadehamid@gmail.com"')
    # .run_commands(
    #     "git clone https://github.com/aadehamid/GPU_Programming_Explainer.git"
    # )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .run_commands(
        "/root/.local/bin/uv venv && . .venv/bin/activate && /root/.local/bin/uv pip install mojo \
                      --index-url https://dl.modular.com/public/nightly/python/simple/ \
                      --prerelease allow"
    )
    # .run_commands("rm -rf /root/GPU_Programming_Explainer")
    # .run_commands("if [ ! -d 'GPU_Programming_Explainer' ]; then git clone https://github.com/aadehamid/GPU_Programming_Explainer.git; fi")
    # .run_commands("cd /root/GPU_Programming_Explainer")
    # .run_commands("git remote set-url origin https://$GITHUB_TOKEN@github.com/aadehamid/GPU_Programming_Explainer.git")
    # .run_commands("git config --global user.name 'aadehamid'")
    # .run_commands('git config --global user.email "aadehamid@gmail.com"')
)


app = modal.App("cuda_images", image=image)


# @app.function(image=image, secrets=[modal.Secret.from_name("github-secret")])
# def get_username():
#     import github

#     g = github.Github(auth=github.Auth.Token(os.environ["GITHUB_TOKEN"]))
#     return g.get_user().login


# @app.function(image=image, secrets=[modal.Secret.from_name("github-secret")])
# def clone_repo(
#     repo_url="https://github.com/aadehamid/GPU_Programming_Explainer.git", branch="main"
# ):
#     assert repo_url.startswith("https://")
#     repo_url_with_creds = repo_url.replace(
#         "https://", "https://" + os.environ["GITHUB_TOKEN"] + "@"
#     )
#     with tempfile.TemporaryDirectory() as dname:
#         print("Cloning", repo_url, "to", dname)
#         git.Repo.clone_from(repo_url_with_creds, dname, branch=branch)
#         return os.listdir(dname)


# @app.function(image=image, secrets=[modal.Secret.from_name("github-secret")])
# def cuda_image_build(
#     repo_url="https://github.com/aadehamid/GPU_Programming_Explainer.git", branch="main"
# ):
#     import github

#     g = github.Github(auth=github.Auth.Token(os.environ["GITHUB_TOKEN"]))
#     assert repo_url.startswith("https://")
#     repo_url_with_creds = repo_url.replace(
#         "https://", "https://" + os.environ["GITHUB_TOKEN"] + "@"
#     )
#     with tempfile.TemporaryDirectory() as dname:
#         print("Cloning", repo_url, "to", dname)
#         repo = git.Repo.clone_from(repo_url_with_creds, dname, branch=branch)

#         # Configure git
#         with repo.config_writer() as config:
#             config.set_value("user", "name", "aadehamid")
#             config.set_value("user", "email", "aadehamid@gmail.com")

#         # Set remote URL with token
#         repo.remote("origin").set_url(repo_url_with_creds)

#         print("Repo files:", os.listdir(dname))
#         print("Github username:", g.get_user().login)
#         return os.listdir(dname)


@app.function(gpu="A100")
def cuda_build_image():
    pass


# @app.local_entrypoint()
# def main(
#     repo_url: str = "https://github.com/aadehamid/GPU_Programming_Explainer.git",
#     branch: str = "main",
# ):
#     # print("Github username:", get_username.remote())
#     # print("Repo files:", clone_repo.remote(repo_url))
#     cuda_image_build.remote(repo_url=repo_url, branch=branch)
