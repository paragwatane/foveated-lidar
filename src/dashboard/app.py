import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import pickle
import sys, os
from sklearn.cluster import DBSCAN
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

st.set_page_config(layout="wide", page_title="Foveated Lidar Map")
st.title("Foveated 2.5D Lidar Mapping — Live Dashboard")

DATAROOT = r"D:\foveated-lidar\data\nuscenes"

REMAP = {0:-1,1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:2,9:1,10:1,11:1,12:1,13:1,
         14:2,15:2,16:2,17:2,18:2,19:2,20:2,21:2,22:2,23:2,
         24:0,25:0,26:0,27:0,28:1,29:1,30:1,31:-1}

CLASS_COLORS = {0: "#66cc66", 1: "#3377dd", 2: "#dd3333"}
CLASS_NAMES = {0: "Terrain", 1: "Static Obstacle", 2: "Dynamic Object"}

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
        dyn_points = points[dynamic_mask][:, :2]
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(dyn_points)
        cluster_ids = clustering.labels_
        dyn_ring_idx = ring_idx[dynamic_mask].copy()
        for cid in set(cluster_ids):
            if cid == -1: continue
            cmask = cluster_ids == cid
            dyn_ring_idx[cmask] = dyn_ring_idx[cmask].min()
        ring_idx[dynamic_mask] = dyn_ring_idx

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
    ax.set_title("Foveated 2.5D Grid (top-down view)")
    return fig

nusc = load_nusc()
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
points = pc.points.T[:, :3]
lidarseg_path = os.path.join(DATAROOT, "lidarseg", "v1.0-mini", lidar_token + "_lidarseg.bin")
raw_labels = np.fromfile(lidarseg_path, dtype=np.uint8)
bucket_labels = np.vectorize(REMAP.get)(raw_labels)

grid, ring_edges, n_valid_points = build_grid_v1(points, bucket_labels, n_angle_bins=n_angle_bins)
t1 = time.time()

latency_ms = (t1-t0)*1000
fps = 1000/latency_ms if latency_ms > 0 else 0

uniform_cells_at_5cm = int(np.pi * 100**2 / (0.05**2))
theoretical_compression = uniform_cells_at_5cm / len(grid)
point_to_cell_ratio = n_valid_points / len(grid)

col1, col2 = st.columns([2,1])
with col1:
    fig = render_grid(grid, ring_edges, n_angle_bins)
    st.pyplot(fig)

with col2:
    st.metric("Frame", f"{frame_idx+1}/{len(sample_tokens)}")
    st.metric("Latency", f"{latency_ms:.1f} ms")
    st.metric("FPS", f"{fps:.1f}")
    st.metric("Raw points", f"{n_valid_points:,}")
    st.metric("Grid cells", f"{len(grid):,}")
    st.metric("Point-to-cell compression", f"{point_to_cell_ratio:.1f}x")
    st.metric("vs theoretical uniform 5cm grid", f"{theoretical_compression:.0f}x")
    st.markdown("**Legend**")
    for cid, name in CLASS_NAMES.items():
        st.markdown(f"<span style='color:{CLASS_COLORS[cid]}'>■</span> {name}", unsafe_allow_html=True)
