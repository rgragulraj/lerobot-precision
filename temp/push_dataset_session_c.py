"""
Push Policy 1b Session C dataset to Hugging Face Hub.
Run: python temp/push_dataset_session_c.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("rgragulraj/policy1_diverse_session_c")
ds.push_to_hub(tags=["so101", "precision", "insertion", "policy1"])

print("Done.")
