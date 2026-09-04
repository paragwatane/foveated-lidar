import numpy as np
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
ring_idx_raw = np.digitize(r, ring_edges) - 1

dynamic_mask = labels == 2
dyn_points = points[dynamic_mask][:, :2]
dyn_ring_raw = ring_idx_raw[dynamic_mask]

clustering = DBSCAN(eps=0.75, min_samples=5).fit(dyn_points)
cluster_ids = clustering.labels_

print("Checking each dynamic cluster for ring-splitting BEFORE v1 fix (i.e. raw v0 assignment):")
split_count = 0
for cid in set(cluster_ids):
    if cid == -1:
        continue
    rings_touched = set(dyn_ring_raw[cluster_ids == cid])
    if len(rings_touched) > 1:
        split_count += 1
        print(f"  Cluster {cid}: SPANS {len(rings_touched)} rings -> {sorted(rings_touched)} (would be split in v0)")

n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
print(f"Total clusters found: {n_clusters}")
print(f"Clusters that WOULD be split by naive v0 ring assignment: {split_count}")
