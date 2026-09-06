import numpy as np
import torch
import torch.nn as nn
import time
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os

DATAROOT = r"D:\foveated-lidar\data\nuscenes"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class PointClassifier(nn.Module):
    def __init__(self, in_dim=4, hidden=64, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x):
        return self.net(x)

t0 = time.time()
ckpt = torch.load("src/segmentation/point_classifier.pt", weights_only=False)
model = PointClassifier().to(device)
model.load_state_dict(ckpt["model"])
model.eval()
mean, std = ckpt["mean"], ckpt["std"]
print(f"Model load: {(time.time()-t0)*1000:.1f} ms")

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
sample = nusc.get('sample', nusc.scene[0]['first_sample_token'])
lidar_token = sample['data']['LIDAR_TOP']
lidar_data = nusc.get('sample_data', lidar_token)
pc_path = os.path.join(DATAROOT, lidar_data['filename'])

t0 = time.time()
pc = LidarPointCloud.from_file(pc_path)
points_full = pc.points.T[:, :4]
print(f"File load: {(time.time()-t0)*1000:.1f} ms")

# Run inference 5 times to see if it speeds up after first call (CUDA warmup)
for i in range(5):
    t0 = time.time()
    X = (points_full - mean) / std
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_t).argmax(1).cpu().numpy()
    print(f"Inference run {i+1}: {(time.time()-t0)*1000:.1f} ms")
