import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
from sklearn.cluster import DBSCAN
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

st.set_page_config(layout="wide", page_title="Foveated Lidar Map")
st.title("Foveated 2.5D Lidar Mapping — Live Dashboard")

DATAROOT = r"D:\foveated-lidar\data\nuscenes"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_COLORS = {0: "#66cc66", 1: "#3377dd", 2: "#dd3333"}
CLASS_NAMES = {0: "Terrain", 1: "Static Obstacle", 2: "Dynamic Object"}

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

@st.cache_resource
def load_model():
    ckpt = torch.load("src/segmentation/point_classifier.pt", weights_only=False)
    model = PointClassifier().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"]

@st.cache_resource
def load_nusc():
    return NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)

def build_grid_v1(points, labels, max_range=100.0, near_range=10.0,
                   fine_res=0.05, coarse_res=0.5, n_angle_bins=128,
                   dbscan_eps=0.75, dbscan_min_samples=5):
    mask = labels >= 0
    points, labels = points[mask], labels[mask]
    x, y, z = points[:,0], points[:,1], points[:,2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    ring_edges = [0.0]
    d = 0.0
    while d < max_range:
        res = fine_res if d < near_range else coarse_res
        d += res
        ring_edges.append(d)
    ring_edges = np.array(ring_edges)

    ring_idx = np.digitize(r, ring_edges) - 1
    angle_idx = ((theta + np.pi) / (2*np.pi) * n_angle_bins).astype(int) % n_angle_bins
    valid = (ring_idx >= 0) & (ring_idx < len(ring_edges)-1)
    points, labels, ring_idx, angle_idx = points[valid], labels[valid], ring_idx[valid], angle_idx[valid]
    z = points[:,2]

    dynamic_mask = labels == 2
    if dynamic_mask.sum() > 0:
        dyn_indices = np.where(dynamic_mask)[0]
        dyn_points_full = points[dyn_indices][:, :2]

        MAX_DBSCAN_POINTS = 3000
        if len(dyn_points_full) > MAX_DBSCAN_POINTS:
            sample_idx = np.random.choice(len(dyn_points_full), MAX_DBSCAN_POINTS, replace=False)
            dyn_points = dyn_points_full[sample_idx]
            dyn_indices_used = dyn_indices[sample_idx]
        else:
            dyn_points = dyn_points_full
            dyn_indices_used = dyn_indices

        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(dyn_points)
        cluster_ids = clustering.labels_
        dyn_ring_idx = ring_idx[dyn_indices_used].copy()
        for cid in set(cluster_ids):
            if cid == -1:
                continue
            cmask = cluster_ids == cid
            dyn_ring_idx[cmask] = dyn_ring_idx[cmask].min()
        ring_idx[dyn_indices_used] = dyn_ring_idx

    cell_keys = ring_idx * n_angle_bins + angle_idx
    unique_keys, inverse = np.unique(cell_keys, return_inverse=True)

    grid = {}
    for i, key in enumerate(unique_keys):
        cm = inverse == i
        majority_class = np.bincount(labels[cm]).argmax()
        grid[int(key)] = {
            "ring": int(key // n_angle_bins), "angle": int(key % n_angle_bins),
            "height": float(z[cm].max()), "class": int(majority_class)
        }
    return grid, ring_edges, len(points)

def render_grid(grid, ring_edges, n_angle_bins=128):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor("white")
    for cell in grid.values():
        ring, angle = cell["ring"], cell["angle"]
        r_in, r_out = ring_edges[ring], ring_edges[ring+1]
        theta_in = (angle/n_angle_bins)*360 - 180
        theta_out = ((angle+1)/n_angle_bins)*360 - 180
        wedge = patches.Wedge((0,0), r_out, theta_in, theta_out, width=r_out-r_in,
                                facecolor=CLASS_COLORS.get(cell["class"], "#999999"),
                                edgecolor="none", alpha=0.85)
        ax.add_patch(wedge)
    ax.set_xlim(-60,60); ax.set_ylim(-60,60)
    ax.set_aspect('equal')
    ax.plot(0,0,'k^',markersize=10)
    ax.set_title("Foveated 2.5D Grid — Model-Predicted Classes (top-down view)")
    return fig

CAMERA_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

def load_camera_images(nusc, sample):
    images = {}
    for cam in CAMERA_CHANNELS:
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        img_path = os.path.join(DATAROOT, cam_data['filename'])
        images[cam] = img_path
    return images

nusc = load_nusc()
model, mean, std = load_model()

_warmup = torch.zeros((1, 4), dtype=torch.float32).to(device)
with torch.no_grad():
    _ = model(_warmup)

scene = nusc.scene[0]
sample_tokens = []
tok = scene['first_sample_token']
while tok:
    sample_tokens.append(tok)
    tok = nusc.get('sample', tok)['next']

st.sidebar.header("Controls")
frame_idx = st.sidebar.slider("Frame", 0, len(sample_tokens)-1, 0)
n_angle_bins = st.sidebar.slider("Angle bins", 32, 256, 128, step=32)

sample = nusc.get('sample', sample_tokens[frame_idx])
lidar_token = sample['data']['LIDAR_TOP']
lidar_data = nusc.get('sample_data', lidar_token)

t0 = time.time()
pc_path = os.path.join(DATAROOT, lidar_data['filename'])
pc = LidarPointCloud.from_file(pc_path)
points_full = pc.points.T[:, :4]
points_xyz = points_full[:, :3]
t_load = time.time()

X = (points_full - mean) / std
X_t = torch.tensor(X, dtype=torch.float32).to(device)
with torch.no_grad():
    bucket_labels = model(X_t).argmax(1).cpu().numpy()
t_infer = time.time()

grid, ring_edges, n_valid_points = build_grid_v1(points_xyz, bucket_labels, n_angle_bins=n_angle_bins)
t1 = time.time()

latency_ms = (t1-t0)*1000
fps = 1000/latency_ms if latency_ms > 0 else 0

uniform_cells_at_5cm = int(np.pi * 100**2 / (0.05**2))
theoretical_compression = uniform_cells_at_5cm / len(grid)
point_to_cell_ratio = n_valid_points / len(grid)

st.sidebar.write(f"Load: {(t_load-t0)*1000:.1f} ms")
st.sidebar.write(f"Inference: {(t_infer-t_load)*1000:.1f} ms")
st.sidebar.write(f"Grid build: {(t1-t_infer)*1000:.1f} ms")

col1, col2 = st.columns([2,1])
with col1:
    fig = render_grid(grid, ring_edges, n_angle_bins)
    st.pyplot(fig)

with col2:
    st.metric("Frame", f"{frame_idx+1}/{len(sample_tokens)}")
    st.metric("Latency (incl. model inference)", f"{latency_ms:.1f} ms")
    st.metric("FPS", f"{fps:.1f}")
    st.metric("Raw points", f"{n_valid_points:,}")
    st.metric("Grid cells", f"{len(grid):,}")
    st.metric("Point-to-cell compression", f"{point_to_cell_ratio:.1f}x")
    st.metric("vs theoretical uniform 5cm grid", f"{theoretical_compression:.0f}x")
    st.markdown("**Legend**")
    for cid, name in CLASS_NAMES.items():
        st.markdown(f"<span style='color:{CLASS_COLORS[cid]}'>■</span> {name}", unsafe_allow_html=True)
    st.caption("Classifications shown are live predictions from the trained neural network, not ground truth.")
    st.markdown("---")
st.subheader("Camera Views — Same Frame, All 6 Directions")
st.caption("Visual reference to verify the LIDAR-based classification above against the real scene.")

cam_images = load_camera_images(nusc, sample)
cols = st.columns(3)
for i, cam in enumerate(CAMERA_CHANNELS):
    with cols[i % 3]:
        st.image(cam_images[cam], caption=cam.replace('_', ' ').title(), use_container_width=True)