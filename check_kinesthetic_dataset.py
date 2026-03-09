from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/so101_kinesthetic_task", root="data/so101_kinesthetic_task")
print(f"Episodes : {dataset.num_episodes}")
print(f"Frames   : {dataset.num_frames}")
print(f"Features : {list(dataset.features.keys())}")
print(f"FPS      : {dataset.fps}")