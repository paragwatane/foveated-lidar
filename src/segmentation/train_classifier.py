import numpy as np
import torch
import torch.nn as nn
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os

DATAROOT = r"D:\foveated-lidar\data\nuscenes"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

def load_all_samples(nusc, sample_tokens):
    all_points, all_labels = [], []
    for tok in sample_tokens:
        sample = nusc.get('sample', tok)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        pc_path = os.path.join(DATAROOT, lidar_data['filename'])
        pc = LidarPointCloud.from_file(pc_path)
        pts = pc.points.T[:, :4]  # x,y,z,intensity
        lidarseg_path = os.path.join(DATAROOT, "lidarseg", "v1.0-mini", lidar_token + "_lidarseg.bin")
        raw = np.fromfile(lidarseg_path, dtype=np.uint8)
        bucket = np.vectorize(REMAP.get)(raw)
        valid = bucket >= 0
        all_points.append(pts[valid])
        all_labels.append(bucket[valid])
    return np.concatenate(all_points), np.concatenate(all_labels)

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)

# Split scenes: first 8 for training, last 2 held out for validation
train_tokens, val_tokens = [], []
for i, scene in enumerate(nusc.scene):
    tok = scene['first_sample_token']
    while tok:
        (train_tokens if i < 8 else val_tokens).append(tok)
        tok = nusc.get('sample', tok)['next']

print(f"Train samples: {len(train_tokens)}, Val samples: {len(val_tokens)}")

print("Loading training data...")
X_train, y_train = load_all_samples(nusc, train_tokens)
print("Loading validation data...")
X_val, y_val = load_all_samples(nusc, val_tokens)

print(f"Train points: {X_train.shape[0]:,}, Val points: {X_val.shape[0]:,}")

# Normalize features
mean, std = X_train.mean(0), X_train.std(0)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

model = PointClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

batch_size = 8192
n_epochs = 15
n_train = X_train_t.shape[0]

print("Training...")
for epoch in range(n_epochs):
    model.train()
    perm = torch.randperm(n_train)
    total_loss = 0
    for i in range(0, n_train, batch_size):
        idx = perm[i:i+batch_size]
        xb, yb = X_train_t[idx], y_train_t[idx]
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 3 == 0 or epoch == n_epochs-1:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).argmax(1)
            val_acc = (val_pred == y_val_t).float().mean().item()
        print(f"Epoch {epoch+1}/{n_epochs} - loss: {total_loss:.2f} - val_acc: {val_acc:.3f}")

torch.save({"model": model.state_dict(), "mean": mean, "std": std}, "src/segmentation/point_classifier.pt")
print("Saved model to src/segmentation/point_classifier.pt")
