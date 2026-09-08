<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/nuScenes-000000?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgo=&logoColor=white" alt="nuScenes"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🔬 Foveated 2.5D LiDAR Mapping</h1>

<p align="center">
  <strong>Bio-inspired, adaptive-resolution LiDAR compression for real-time autonomous driving perception</strong>
</p>

<p align="center">
  <em>High resolution where it matters. Efficient everywhere else.</em>
</p>

---

## 📌 Overview

**Foveated 2.5D LiDAR Mapping** applies the biological concept of [foveated vision](https://en.wikipedia.org/wiki/Fovea_centralis) — where the eye's center has far greater acuity than its periphery — to 3D LiDAR point cloud processing for autonomous driving.

The system constructs an **adaptive-resolution polar grid** around the ego-vehicle:

| Zone | Range | Cell Size | Purpose |
|------|-------|-----------|---------|
| **Foveal** (near-field) | 0 – 10 m | 5 cm | Fine-grained perception of nearby dynamic objects |
| **Peripheral** (far-field) | 10 – 100 m | 50 cm | Coarse but sufficient representation of distant terrain |

This achieves **massive compression** (typically **10–30×** point-to-cell reduction, and **~12,500×** vs. a uniform 5 cm grid) while preserving critical detail near the vehicle where reaction time matters most.

---

## 🏗️ Architecture

The project follows a modular four-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FOVEATED LIDAR PIPELINE                         │
│                                                                        │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐ │
│   │   (A)    │    │     (B)      │    │    (C)      │    │   (D)    │ │
│   │ nuScenes │───▶│ Segmentation │───▶│ Grid Engine │───▶│Dashboard │ │
│   │   Data   │    │   (PyTorch)  │    │  (Foveated) │    │(Streamlit│ │
│   │  Loader  │    │              │    │             │    │   + Viz) │ │
│   └──────────┘    └──────────────┘    └─────────────┘    └──────────┘ │
│                                                                        │
│   Raw LiDAR ──▶ Semantic Labels ──▶ Compressed Grid ──▶ Visualization │
│   (34,688 pts)  (3-class)          (~2,500 cells)       (Live metrics)│
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage Descriptions

| Stage | Module | Description |
|-------|--------|-------------|
| **(A) Data Ingestion** | `src/segmentation/load_nuscenes_sample.py` | Loads raw LiDAR point clouds and per-point semantic labels from the nuScenes Mini dataset |
| **(B) Semantic Segmentation** | `src/segmentation/train_classifier.py` | 3-layer MLP point classifier (4D input → 3 classes: *terrain*, *static obstacle*, *dynamic object*) trained with class-weighted cross-entropy loss |
| **(C) Grid Engine** | `src/grid_engine/grid_v0.py`, `grid_v1.py` | Builds the foveated polar grid. **v1** adds DBSCAN clustering to snap dynamic-object clusters to the finest ring they touch, preventing resolution-boundary artifacts |
| **(D) Dashboard** | `src/dashboard/app.py` | Interactive Streamlit app with live inference, top-down grid visualization, latency/FPS metrics, and synchronized 6-camera views |

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.8+
- **CUDA** (optional, for GPU-accelerated inference)
- **nuScenes Mini** dataset ([download here](https://www.nuscenes.org/nuscenes#download))

### Installation

```bash
# Clone the repository
git clone https://github.com/paragwatane/foveated-lidar.git
cd foveated-lidar

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install numpy torch scikit-learn matplotlib streamlit nuscenes-devkit
```

### Data Setup

Download the **nuScenes Mini** split and place it under:

```
data/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    ├── lidarseg/
    ├── v1.0-mini/
    └── ...
```

> **Note:** Update the `DATAROOT` variable in source files if your data lives elsewhere.

---

## ⚡ Usage

### 1. Train the Segmentation Model

```bash
python src/segmentation/train_classifier.py
```

Trains a 3-layer MLP on the nuScenes Mini dataset with inverse-frequency class weighting. Outputs `src/segmentation/point_classifier.pt`.

### 2. Evaluate Model Accuracy

```bash
# Per-class precision & recall
python src/segmentation/per_class_accuracy.py

# Accuracy by distance band (0–10m, 10–25m, 25–50m, 50–100m)
python src/segmentation/accuracy_by_distance.py
```

### 3. Run the Grid Engine

```bash
# v0: Basic foveated grid
python src/grid_engine/grid_v0.py

# v1: With DBSCAN dynamic-object clustering
python src/grid_engine/grid_v1.py
```

### 4. Launch the Live Dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard provides:
- 🗺️ **Top-down foveated grid visualization** with color-coded semantic classes
- 📊 **Real-time metrics**: latency, FPS, compression ratio
- 🎛️ **Interactive controls**: frame slider, adjustable angle bins
- 📷 **Synchronized 6-camera views** for visual verification

---

## 📁 Project Structure

```
foveated-lidar/
│
├── src/
│   ├── segmentation/                   # Stage A + B
│   │   ├── load_nuscenes_sample.py     # Data ingestion from nuScenes
│   │   ├── train_classifier.py         # MLP training pipeline
│   │   ├── per_class_accuracy.py       # Per-class evaluation metrics
│   │   ├── accuracy_by_distance.py     # Distance-band accuracy analysis
│   │   ├── diagnose_dbscan.py          # DBSCAN parameter diagnostics
│   │   ├── diagnose_speed.py           # Inference speed benchmarking
│   │   └── point_classifier.pt         # Trained model checkpoint
│   │
│   ├── grid_engine/                    # Stage C
│   │   ├── grid_v0.py                  # Baseline foveated grid
│   │   └── grid_v1.py                  # + DBSCAN cluster-aware grid
│   │
│   ├── dashboard/                      # Stage D
│   │   └── app.py                      # Streamlit live dashboard
│   │
│   └── metrics/                        # Metrics logging (planned)
│
├── data/                               # nuScenes dataset (gitignored)
├── logs/                               # Runtime logs (gitignored)
├── CONTRACT.md                         # Inter-module data contracts
├── .gitignore
└── README.md
```

---

## 🔧 Technical Details

### Segmentation Model

| Property | Value |
|----------|-------|
| Architecture | 3-layer MLP (4 → 64 → 64 → 64 → 3) |
| Input Features | x, y, z, intensity |
| Output Classes | Terrain (0), Static Obstacle (1), Dynamic Object (2) |
| Loss Function | Cross-entropy with inverse-frequency class weights |
| Training Data | nuScenes Mini (8 scenes train / 2 scenes val) |
| Batch Size | 8,192 |
| Optimizer | Adam (lr=1e-3) |

### Foveated Grid (v1)

| Property | Value |
|----------|-------|
| Near-field resolution | 5 cm (0 – 10 m) |
| Far-field resolution | 50 cm (10 – 100 m) |
| Angular bins | 128 (configurable 32 – 256) |
| Dynamic clustering | DBSCAN (ε=0.75, min_samples=5) |
| Cluster snapping | Entire cluster → finest ring touched |

### Data Contracts

All inter-module data formats are defined in [`CONTRACT.md`](CONTRACT.md):

- **Segmentation → Grid Engine**: `.npz` files with `points` (N×3, float32) and `labels` (N, int)
- **Grid Engine → Dashboard**: List of cell dicts with `ring`, `angle_bin`, `height`, `class`
- **Metrics**: CSV with `frame_id`, `latency_ms`, `num_points`, `num_cells`, `miou_*`

---

## 🧪 Diagnostics & Benchmarking

```bash
# Benchmark inference speed (5 runs with CUDA warmup)
python src/segmentation/diagnose_speed.py

# Analyze DBSCAN clustering time on dynamic-class points
python src/segmentation/diagnose_dbscan.py
```

---

## 🤝 Contributing

This is a team project with modular ownership:

| Module | Owner |
|--------|-------|
| Segmentation Pipeline | Person A |
| Grid Engine | Person B |
| Dashboard | Person C |
| Metrics | Person D |

To contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m "Add your feature"`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

---

## 📚 References

- **nuScenes Dataset**: Caesar, H. et al. *"nuScenes: A multimodal dataset for autonomous driving."* CVPR 2020. [[Paper]](https://arxiv.org/abs/1903.11027) [[Website]](https://www.nuscenes.org/)
- **Foveated Vision**: Inspired by biological foveation — the human eye's variable-acuity sampling strategy
- **DBSCAN**: Ester, M. et al. *"A density-based algorithm for discovering clusters in large spatial databases with noise."* KDD 1996.

---

<p align="center">
  <sub>Built with ❤️ for real-time autonomous driving perception</sub>
</p>