import numpy as np
import torch
import torch.nn as nn
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os

DATAROOT = r"D:\foveated-lidar\data\nuscenes"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REMAP = {0:-1,1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:2,9:1,10:1,11:1,12:1,13:1,
         14:2,15:2,16:2,17:2,18:2,19:2,20:2,21:2,22:2,23:2,
         24:0,25:0,26:0,27:0,28:1,29:1,30:1,31:-1}

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

ckpt = torch.load("src/segmentation/point_classifier.pt")
model = PointClassifier().to(device)
model.load_state_dict(ckpt["model"])
model.eval()
mean, std = ckpt["mean"], ckpt["std"]

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)

val_tokens = []
for i, scene in enumerate(nusc.scene):
    if i < 8:
        continue
    tok = scene['first_sample_token']
    while tok:
        val_tokens.append(tok)
        tok = nusc.get('sample', tok)['next']

all_points, all_labels = [], []
for tok in val_tokens:
    sample = nusc.get('sample', tok)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    pc_path = os.path.join(DATAROOT, lidar_data['filename'])
    pc = LidarPointCloud.from_file(pc_path)
    pts = pc.points.T[:, :4]
    lidarseg_path = os.path.join(DATAROOT, "lidarseg", "v1.0-mini", lidar_token + "_lidarseg.bin")
    raw = np.fromfile(lidarseg_path, dtype=np.uint8)
    bucket = np.vectorize(REMAP.get)(raw)
    valid = bucket >= 0
    all_points.append(pts[valid])
    all_labels.append(bucket[valid])

points = np.concatenate(all_points)
labels = np.concatenate(all_labels)

X = (points - mean) / std
X_t = torch.tensor(X, dtype=torch.float32).to(device)

with torch.no_grad():
    preds = model(X_t).argmax(1).cpu().numpy()

r = np.sqrt(points[:,0]**2 + points[:,1]**2)

bands = [(0,10), (10,25), (25,50), (50,100)]
print("Accuracy by distance band:")
print(f"{'Range':<12}{'Points':<12}{'Accuracy':<10}")
for lo, hi in bands:
    mask = (r >= lo) & (r < hi)
    if mask.sum() == 0:
        continue
    acc = (preds[mask] == labels[mask]).mean()
    label = f"{lo}-{hi}m"
    print(f"{label:<12}{mask.sum():<12,}{acc:.3f}")

overall_acc = (preds == labels).mean()
print("")
print(f"Overall accuracy: {overall_acc:.3f}")