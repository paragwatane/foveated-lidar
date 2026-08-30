Segmentation output (A → B):
  points: np.ndarray, shape (N, 3), float32, (x, y, z)
  labels: np.ndarray, shape (N,), int, values in {0=terrain, 1=static, 2=dynamic}
  saved as: frame_<id>.npz with keys "points" and "labels"

Grid engine output (B → C):
  list of cell dicts:
    { "ring": int, "angle_bin": int, "height": float, "class": int }

Metrics log format (all → D):
  CSV columns: frame_id, latency_ms, num_points, num_cells, miou_terrain, miou_static, miou_dynamic
