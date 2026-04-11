"""
Push the trained Policy 1 Session A model to Hugging Face Hub.
Run: python temp/push_model_session_a.py
"""

from huggingface_hub import HfApi

MODEL_PATH = "/home/rgragulraj/models/policy1_session_a"
REPO_ID = "rgragulraj/policy1_session_a"

api = HfApi()

print(f"Creating repo {REPO_ID} (if not exists)...")
api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

print(f"Uploading {MODEL_PATH} ...")
api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"Done. Model live at: https://huggingface.co/{REPO_ID}")
