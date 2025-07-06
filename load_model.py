from huggingface_hub import snapshot_download

hf_token = "your_hugging_face_token_here"

local_dir = "./mistral-7b-instruct-local"

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir=local_dir,
    use_auth_token=hf_token,
    repo_type="model",
    resume_download=True,
    local_dir_use_symlinks=False  
)

print(f"? Model downloaded successfully to: {local_dir}")
