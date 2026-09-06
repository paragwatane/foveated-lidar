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

raw = np.fromfile(os.path.join(DATAROOT, "lidarseg", "v1.0-mini", lidar_token + "_lidarseg.bin"), dtype=np.uint8)
true_labels = np.vectorize(REMAP.get)(raw)

X = (points_full - mean) / std
X_t = torch.tensor(X, dtype=torch.float32).to(device)
with torch.no_grad():
    preds = model(X_t).argmax(1).cpu().numpy()

for cls, name in [(0,"Terrain"), (1,"Static"), (2,"Dynamic")]:
    true_mask = true_labels == cls
    pred_mask = preds == cls
    tp = np.sum(true_mask & pred_mask)
    print(f"{name}: true count={np.sum(true_mask)}, predicted count={np.sum(pred_mask)}, correct={tp}, precision={tp/max(np.sum(pred_mask),1):.2f}, recall={tp/max(np.sum(true_mask),1):.2f}")
