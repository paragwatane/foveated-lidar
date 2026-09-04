import numpy as np
import pickle
from sklearn.cluster import DBSCAN

data = np.load("sample_frame_0_bucketed.npz")
points, labels = data["points"], data["labels"]
mask = labels >= 0
points, labels = points[mask], labels[mask]
x, y = points[:,0], points[:,1]
r = np.sqrt(x**2 + y**2)

fine_res, coarse_res, near_range, max_range = 0.05, 0.5, 10.0, 100.0
ring_edges = [0.0]
d = 0.0
while d < max_range:
    res = fine_res if d < near_range else coarse_res
    d += res
    ring_edges.append(d)
ring_edges = np.array(ring_edges)
ring_idx = np.digitize(r, ring_edges) - 1

dynamic_mask = labels == 2
dyn_points = points[dynamic_mask][:, :2]
dyn_ring = ring_idx[dynamic_mask].copy()

clustering = DBSCAN(eps=0.75, min_samples=5).fit(dyn_points)
cluster_ids = clustering.labels_

# Apply the SAME fix as grid_v1.py
for cid in set(cluster_ids):
    if cid == -1:
        continue
    cmask = cluster_ids == cid
    finest_ring = dyn_ring[cmask].min()
    dyn_ring[cmask] = finest_ring

print("Checking each dynamic cluster AFTER v1 fix:")
split_count = 0
for cid in set(cluster_ids):
    if cid == -1:
        continue
    rings_touched = set(dyn_ring[cluster_ids == cid])
    if len(rings_touched) > 1:
        split_count += 1
        print(f"  Cluster {cid}: still spans {len(rings_touched)} rings (BUG)")

print(f"Clusters still split after v1 fix: {split_count} (should be 0)")
