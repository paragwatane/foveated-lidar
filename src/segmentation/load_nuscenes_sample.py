import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os

DATAROOT = r"D:\foveated-lidar\data\nuscenes"

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=True)

scene = nusc.scene[0]
sample_token = scene['first_sample_token']
sample = nusc.get('sample', sample_token)

lidar_token = sample['data']['LIDAR_TOP']
lidar_data = nusc.get('sample_data', lidar_token)

pc_path = os.path.join(DATAROOT, lidar_data['filename'])
pc = LidarPointCloud.from_file(pc_path)
points = pc.points.T[:, :3]

print("Points shape:", points.shape)

lidarseg_path = os.path.join(DATAROOT, "lidarseg", "v1.0-mini", lidar_token + "_lidarseg.bin")
labels = np.fromfile(lidarseg_path, dtype=np.uint8)

print("Labels shape:", labels.shape)
print("Unique classes present:", np.unique(labels))

assert points.shape[0] == labels.shape[0], "Mismatch between points and labels!"

np.savez("sample_frame_0.npz", points=points.astype(np.float32), labels=labels.astype(np.int64))
print("Saved sample_frame_0.npz")