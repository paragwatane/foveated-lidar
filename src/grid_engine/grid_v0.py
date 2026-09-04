import numpy as np

def build_foveated_grid(points, labels, max_range=100.0, near_range=10.0,
                          fine_res=0.05, coarse_res=0.5, n_angle_bins=360):
    mask = labels >= 0
    points = points[mask]
    labels = labels[mask]

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
    ring_idx, angle_idx, z, labels = ring_idx[valid], angle_idx[valid], z[valid], labels[valid]

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
    grid, ring_edges = build_foveated_grid(points, labels)
    print(f"Total points: {len(points)}")
    print(f"Total grid cells: {len(grid)}")
    print(f"Compression ratio: {len(points)/len(grid):.1f}x fewer cells than points")
