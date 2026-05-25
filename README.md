# 🌊 Flood Detection via Satellite Image Segmentation

> Deep learning pipeline for pixel-level flood area detection using multi-channel SAR and multispectral satellite imagery — built for the **ANRF AISEHack @ IIIT Hyderabad (IBM Theme 1)**

---

## 🏆 Competition Results

| Phase | Rank | Score | Total Teams |
|-------|------|-------|-------------|
| Phase 1 — Binary Segmentation | **38th** | 0.712 IoU | 103 teams |
| Phase 2 — 3-Class Segmentation | **14th** | 0.1865 | 57 shortlisted teams |

---

## 📌 Problem Statement

Given satellite imagery (SAR + multispectral bands) from IBM PAIRS, the goal is to detect and segment flood-affected areas at the pixel level. Phase 2 extended this to 3-class classification:
- `0` — Dry land
- `1` — Flood water
- `2` — Flooded vegetation / land

---

## 🏗️ Architecture

```
Input (Multi-channel)
    ├── SAR bands (VV, VH)
    ├── Multispectral bands (RGB, NIR)
    └── Derived indices (NDWI, NDVI, VV/VH ratio)
         ↓
    UNet++ Encoder-Decoder
    ├── Backbone: EfficientNet-B5 (pretrained ImageNet)
    ├── Attention: scSE (Spatial + Channel Squeeze & Excitation)
    └── Skip connections with dense feature aggregation
         ↓
    Loss: Focal Loss + Dice Loss (combined)
         ↓
    Test-Time Augmentation (TTA)
         ↓
    Segmentation Mask Output
```

---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, segmentation-models-pytorch |
| Data Processing | NumPy, Pandas, Rasterio |
| Visualization | Matplotlib, Seaborn |
| Training | Kaggle Notebooks (GPU T4/P100) |
| Dataset | IBM PAIRS (SAR + Sentinel-2 imagery) |

---

## 🚀 Key Techniques

- **Multi-channel input engineering** — Combined SAR polarization bands with multispectral data and computed indices (NDWI, NDVI) to enrich model inputs
- **UNet++ with EfficientNet-B5** — Dense skip connections improve gradient flow and capture fine-grained flood boundaries
- **scSE Attention** — Channel and spatial recalibration focuses learning on flood-relevant features
- **Focal + Dice Loss** — Handles severe class imbalance (flood pixels << background pixels)
- **Test-Time Augmentation** — Horizontal/vertical flips averaged at inference for more robust predictions
- **Phase 2 Ensemble** — Multiple model predictions merged to improve 3-class segmentation stability

---

## 📂 Project Structure

```
flood-detection/
├── notebooks/
│   ├── phase1_binary_segmentation.ipynb
│   └── phase2_multiclass_segmentation.ipynb
├── src/
│   ├── dataset.py          # Custom PyTorch Dataset class
│   ├── model.py            # UNet++ model definition
│   ├── loss.py             # Focal + Dice loss
│   ├── train.py            # Training loop
│   └── predict.py          # Inference + TTA
├── requirements.txt
└── README.md
```

---

## 📊 Results & Learnings

- Ensembling weak models degraded performance — model quality matters more than quantity
- Validation IoU on small datasets is unreliable as a leaderboard proxy — overfitting risk is high
- Multi-channel input (SAR + derived indices) consistently outperformed RGB-only baselines
- TTA gave a consistent +0.005–0.01 IoU improvement across phases

---

## ⚙️ Setup & Usage

```bash
git clone https://github.com/jcreddy-6/flood-detection
cd flood-detection
pip install -r requirements.txt

# Run inference on a satellite image
python src/predict.py --input path/to/image.tif --output prediction_mask.png
```

---

## 📜 Competition

- **Event:** ANRF AISEHack — Theme 1: Flood Detection (IBM)
- **Organized by:** IIIT Hyderabad in collaboration with IBM
- **Dataset:** IBM PAIRS Geoscope satellite imagery platform

---

## 👤 Author

**Thigulla Jhansi Chandra Reddy**
2nd Year B.Tech CSE — Anurag University, Hyderabad
[LinkedIn](https://www.linkedin.com/in/jhansi-chandra-reddy-a14b15382/) | [GitHub](https://github.com/jcreddy-6)
