import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.cluster import DBSCAN
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os

DATAROOT = r"D:\foveated-lidar\data\nuscenes"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

ckpt = torch.load("src/segmentation/point_classifier.pt", weights_only=False)
model = PointClassifier().to(device)
model.load_state_dict(ckpt["model"])
model.eval()
mean, std = ckpt["mean"], ckpt["std"]

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
sample = nusc.get('sample', nusc.scene[0]['first_sample_token'])
lidar_token = sample['data']['LIDAR_TOP']
lidar_data = nusc.get('sample_data', lidar_token)
pc_path = os.path.join(DATAROOT, lidar_data['filename'])
pc = LidarPointCloud.from_file(pc_path)
points_full = pc.points.T[:, :4]

X = (points_full - mean) / std
X_t = torch.tensor(X, dtype=torch.float32).to(device)
with torch.no_grad():
    preds = model(X_t).argmax(1).cpu().numpy()

print("Total points:", len(preds))
print("Predicted terrain:", np.sum(preds==0))
print("Predicted static:", np.sum(preds==1))
print("Predicted dynamic:", np.sum(preds==2))

dyn_points = points_full[preds==2][:, :2]
t0 = time.time()
clustering = DBSCAN(eps=0.75, min_samples=5).fit(dyn_points)
print(f"DBSCAN time: {(time.time()-t0)*1000:.1f} ms for {len(dyn_points)} points")
