import numpy as np
from sklearn.cluster import DBSCAN

def build_foveated_grid_v1(points, labels, max_range=100.0, near_range=10.0,
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

    # --- Cluster dynamic-object points (class==2) to keep them intact ---
    dynamic_mask = labels == 2
    if dynamic_mask.sum() > 0:
        dyn_points = points[dynamic_mask][:, :2]  # x,y only for clustering
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(dyn_points)
        cluster_ids = clustering.labels_  # -1 = noise, else cluster id

        dyn_ring_idx = ring_idx[dynamic_mask].copy()
        dyn_angle_idx = angle_idx[dynamic_mask].copy()

        # For each real cluster (not noise), snap all its points to the FINEST ring touched
        for cid in set(cluster_ids):
            if cid == -1:
                continue
            cmask = cluster_ids == cid
            finest_ring = dyn_ring_idx[cmask].min()
            dyn_ring_idx[cmask] = finest_ring  # whole cluster -> finest ring it touches

        ring_idx[dynamic_mask] = dyn_ring_idx
        angle_idx[dynamic_mask] = dyn_angle_idx

    # --- Bin into cells (same as v0 from here) ---
    cell_keys = ring_idx * n_angle_bins + angle_idx
    unique_keys, inverse = np.unique(cell_keys, return_inverse=True)

    grid = {}
    for i, key in enumerate(unique_keys):
        cell_mask = inverse == i
        cell_z = z[cell_mask]
        cell_labels = labels[cell_mask]
        majority_class = np.bincount(cell_labels).argmax()
        grid[int(key)] = {
            "ring": int(key // n_angle_bins),
            "angle": int(key % n_angle_bins),
            "height": float(cell_z.max()),
            "class": int(majority_class)
        }
    return grid, ring_edges

if __name__ == "__main__":
    data = np.load("sample_frame_0_bucketed.npz")
    points, labels = data["points"], data["labels"]

    grid_v0_style, _ = build_foveated_grid_v1(points, labels, dbscan_eps=0.75, dbscan_min_samples=5)
    print(f"Total points: {len(points)}")
    print(f"Total grid cells (v1): {len(grid_v0_style)}")
    print(f"Compression ratio: {len(points)/len(grid_v0_style):.1f}x")

    import pickle
    with open("grid_v1_output.pkl", "wb") as f:
        pickle.dump({"grid": grid_v0_style, "ring_edges": _}, f)
    print("Saved grid_v1_output.pkl")
