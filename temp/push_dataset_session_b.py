"""
Push Policy 1b Session B dataset to Hugging Face Hub.
Run: python temp/push_dataset_session_b.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("rgragulraj/policy1_diverse_session_b")
ds.push_to_hub(tags=["so101", "precision", "insertion", "policy1"])

print("Done.")
