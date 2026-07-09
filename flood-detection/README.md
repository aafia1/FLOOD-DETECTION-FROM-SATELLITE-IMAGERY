# 🌊 Flood Detection from Satellite Imagery

Pixel-level flood segmentation from aerial/satellite imagery using **U-Net**, progressed from a scratch-built encoder to a **U-Net + EfficientNetB4** transfer-learning architecture.

Built as a semester project for **Deep Learning (Computer Vision)**.

---

## 📌 Overview

Floods account for roughly 40% of all natural disasters worldwide, and rapid, accurate flood mapping is critical for disaster response. This project builds an automated, pixel-level flood detection system in two progressive stages:

1. **Stage 1 — Scratch U-Net**: A U-Net segmentation model built entirely from scratch in TensorFlow/Keras. Solved a real overfitting problem (caused by only 290 training images) using a 4x data augmentation pipeline.
2. **Stage 2 — U-Net + EfficientNetB4 (Transfer Learning)**: Replaced the scratch encoder with an ImageNet-pretrained EfficientNetB4 backbone, trained with a two-phase (frozen → fine-tune) strategy for stronger performance.

Both models are benchmarked against **Soebroto et al. (2025)**, who reported a best IoU of 0.767 on the same dataset using U-Net + MobileNet.

## 📊 Results

| Model | Dice Score | IoU Score |
|---|---|---|
| Scratch U-Net (no augmentation) | — | 0.409 (val, early-stopped) |
| **Scratch U-Net (with 4x augmentation)** | **0.835** | **0.762** |
| U-Net + EfficientNetB4 (2-phase transfer learning) | See notebook output | See notebook output |
| Soebroto et al. (2025) — U-Net + MobileNet (benchmark) | 0.866 (F1) | 0.767 |

> Run the notebooks to reproduce exact EfficientNetB4 test-set numbers — they depend on your training run and are printed at the end of `02_efficientnetb4_unet.ipynb`.

## 🗂️ Dataset

- **Flood Area Segmentation dataset** — 290 aerial RGB images with binary flood masks
- Roughly 40.7% flood pixels (mild class imbalance)
- Split: 80% train / 10% val / 10% test (with 4x augmentation applied to training set only)
- Not included in this repo due to size — see [Setup](#-setup) for how to add it

## 🏗️ Architecture

**Stage 1 — Scratch U-Net**
- Classic encoder–decoder with skip connections, built from scratch
- Trained on 0–1 normalized images
- Loss: combined Binary Cross-Entropy + Dice Loss

**Stage 2 — U-Net + EfficientNetB4**
- ImageNet-pretrained EfficientNetB4 encoder (via `segmentation-models`) + U-Net decoder
- Images kept in 0–255 range, normalized using EfficientNet's own preprocessing function (applied *after* augmentation)
- **Phase 1**: backbone frozen, decoder trained for 8 epochs (LR = 1e-3)
- **Phase 2**: full network fine-tuned for up to 30 epochs (LR = 1e-5)

## 🧰 Tech Stack

- Python 3, TensorFlow / Keras
- `segmentation-models` (EfficientNetB4 backbone + preprocessing)
- OpenCV, NumPy, scikit-learn, Matplotlib
- Google Colab (notebooks assume Colab + Google Drive mount by default)

## 📁 Repository Structure

```
flood-detection/
├── notebooks/
│   ├── 01_scratch_unet.ipynb          # Stage 1: U-Net from scratch + augmentation
│   └── 02_efficientnetb4_unet.ipynb    # Stage 2: U-Net + EfficientNetB4 transfer learning
├── report/
│   └── Flood_Detection_Report.docx     # Full written report (intro, literature review, methodology, results, discussion)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ⚙️ Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/flood-detection.git
   cd flood-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get the dataset**
   The notebooks expect a folder structure like:
   ```
   data/
   ├── Image/   # satellite/aerial RGB images
   └── Mask/    # corresponding binary flood masks
   ```
   Download the *Flood Area Segmentation* dataset (e.g. from Kaggle) and place it under `data/`.

4. **Update paths**
   The notebooks were originally written for Google Colab and mount Google Drive with paths like:
   ```python
   IMAGE_PATH = '/content/drive/MyDrive/flood_detection/data/Image/'
   MASK_PATH  = '/content/drive/MyDrive/flood_detection/data/Mask/'
   ```
   If running locally or on a different platform, update these paths (and remove/skip the `drive.mount(...)` cell) to point at your local `data/` folder.

5. **Run the notebooks in order**
   - `notebooks/01_scratch_unet.ipynb` — trains and evaluates the scratch U-Net baseline
   - `notebooks/02_efficientnetb4_unet.ipynb` — trains and evaluates the EfficientNetB4 transfer-learning model

## 📄 Report

The full write-up — literature review, methodology, results, comparative analysis, discussion, and future work — is available in [`report/Flood_Detection_Report.docx`](report/Flood_Detection_Report.docx).

## 🔭 Future Work

- Incorporate larger, more diverse datasets (e.g. Sen1Floods11, Copernicus Emergency Management imagery, NASA Earth Observatory data)
- Use SAR imagery instead of RGB for cloud-penetrating, day/night flood mapping
- Move from binary segmentation to flood depth/severity estimation
- Evaluate cross-dataset generalization

## 📚 Key References

- Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
- Tan & Le (2019) — EfficientNet: Rethinking Model Scaling for CNNs
- Soebroto et al. (2025) — Comparative study of U-Net/SegNet/LinkNet backbones on the Flood Area Segmentation dataset (primary benchmark used in this project)

Full reference list is in the report.

## 📜 License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.
