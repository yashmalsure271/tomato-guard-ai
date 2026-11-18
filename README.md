# TomatoGuard AI

AI-assisted tomato leaf diagnostics from a single image. TomatoGuard AI is a
Streamlit web application built around a 10-class ConvNeXt-Tiny classifier,
with confidence scores, Grad-CAM attention maps, batch CSV export and a PDF
report. It is the deployment companion to an IA-2 comparative CNN study.

> **Screening aid, not a diagnosis.** Always confirm plant health decisions with
> a qualified local agronomist or plant clinic.

## What is in the app

- **Home** — project overview, dataset facts and the original four-model comparison.
- **Prediction** — upload or take a photo, review top-3 confidence, inspect Grad-CAM,
  download a PDF report, or score a batch of images and export CSV.
- **Dashboard** — deployed checkpoint metrics, training curves, confusion matrix,
  per-class scores, ROC curves, error analysis and the preserved original run.
- **About** — methodology, provenance, limitations, privacy and credits.

The live model is **ConvNeXt-Tiny** because it achieved the best test accuracy in
the original notebook run (99.70%). The comparison table also includes LDAMNet,
ResNet50 and EfficientNetV2-S.

## Results

### Original notebook run

All four models used the same stratified 70/15/15 split (seed 42) and the
1,650-image held-out test split.

| Model | Parameters | Accuracy | F1 | AUC | Training time |
| --- | ---: | ---: | ---: | ---: | ---: |
| LDAMNet | 1.51M | 95.45% | 95.43% | 0.9988 | 455s |
| ResNet50 | 23.53M | 99.21% | 99.21% | 0.9992 | 805s |
| EfficientNetV2-S | 20.19M | 99.52% | 99.52% | 0.9994 | 909s |
| **ConvNeXt-Tiny** | **27.83M** | **99.70%** | **99.70%** | **0.9999** | **1,340s** |

These values are preserved from the notebook's saved outputs and are displayed
separately from the deployed model in the Dashboard.

### Deployed checkpoint

The original `.pth` files were not preserved with the notebook. The deployed
checkpoint was retrained from the same notebook pipeline on Apple MPS hardware
(10 epochs, Adam 1e-4, batch size 32, stratified 70/15/15 split, seed 42).

| Metric | Value |
| --- | ---: |
| Test accuracy | 99.82% |
| Weighted precision | 99.82% |
| Weighted recall | 99.82% |
| Weighted F1 | 99.82% |
| ROC-AUC (weighted OVR) | 0.9999 |
| Best validation accuracy | 99.64% (epoch 9) |
| Checkpoint size | 55.7 MB |
| Storage | float16, loaded into a float32 model |
| Misclassified test images | 3 / 1,650 |

The deployed checkpoint slightly exceeds the original ConvNeXt-Tiny test
accuracy (99.82% vs. 99.70%). Both numbers are reported side by side in the
Dashboard, and the app never substitutes one for the other. The full metrics are
stored in `artifacts/test_metrics.json`.

The checkpoint is stored in float16 so that it stays below GitHub's 100 MB file
limit; it is loaded into a float32 model at runtime. `artifacts/fp16_equivalence.json`
confirms the float16 file produced identical predictions to the float32 weights
on all 1,650 test images.

## Dataset

- Source: [Project-AgML/tomato_leaf_disease](https://huggingface.co/datasets/Project-AgML/tomato_leaf_disease)
- 11,000 RGB images, 1,100 per class, 256 × 256 pixels
- 10 classes: bacterial spot, early blight, healthy, late blight, leaf mold,
  mosaic virus, septoria leaf spot, spider mites, target spot, yellow leaf curl virus
- Split: 7,700 train / 1,650 validation / 1,650 test, stratified, seed 42

Check the dataset card for license and usage terms.

## Method

1. Load the dataset with Hugging Face `datasets`.
2. Create one stratified 70/15/15 split using seed 42.
3. Train each model with ImageNet initialization, Adam (1e-4), batch size 32,
   cross-entropy loss and 10 epochs; keep the best validation checkpoint.
4. Evaluate on the test split with accuracy, weighted precision/recall/F1 and
   one-vs-rest AUC.
5. Deploy ConvNeXt-Tiny with 224 × 224 resize and ImageNet normalization, and
   add Grad-CAM explanations.

The paper that inspired the LDAMNet baseline is:

> Zhang, E., Zhang, N., Li, F. and Lv, C. (2024). A lightweight dual-attention
> network for tomato leaf disease identification. *Frontiers in Plant Science*.
> https://doi.org/10.3389/fpls.2024.1420584

LDAMNet in this project is a paper-inspired implementation of channel and
spatial attention blocks, not a byte-for-byte reproduction of the published model.

## Project structure

```text
classify/
├── app.py                     # Streamlit entry point and navigation
├── views/
│   ├── home.py
│   ├── prediction.py
│   ├── dashboard.py
│   └── about.py
├── utils/
│   ├── config.py              # paths and project constants
│   ├── constants.py           # class names, display names, model settings
│   ├── model.py               # cached ConvNeXt-Tiny loader
│   ├── inference.py           # preprocessing, single and batch prediction
│   ├── gradcam.py             # Grad-CAM for ConvNeXt-Tiny
│   ├── pdf_report.py          # fpdf2 prediction report
│   ├── disease_info.py        # class descriptions and management notes
│   ├── artifacts.py           # JSON artifact loaders
│   ├── ui.py                  # CSS injection and shared UI components
│   └── ...
├── model/
│   ├── convnext_tiny_best.pth # deployed float16 checkpoint
│   └── model_card.json
├── artifacts/                 # metrics, history, ROC, errors, notebook evidence
├── assets/                    # logo, samples, evidence images, styles.css
├── scripts/
│   ├── prepare_assets.py
│   └── train_convnext.py
├── notebook/                  # original Colab notebook, untouched
├── requirements.txt           # CPU deployment dependencies
└── requirements-dev.txt       # local app + training dependencies
```

## Run locally

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`.

The app needs `model/convnext_tiny_best.pth`. If it is missing, the Prediction
page explains that no trained checkpoint is available rather than displaying a
random model.

## Retrain

Training downloads the dataset and ImageNet ConvNeXt-Tiny weights on first use.

```bash
export HF_HOME="$PWD/.cache/huggingface"
python scripts/train_convnext.py --workers 0 --epochs 10
```

On Apple silicon, keep `--workers 0`; Python 3.12 multiprocessing can fail with
`DataLoader worker ... exited unexpectedly` on macOS. Training on an M4 MPS
device takes roughly 75 to 90 minutes. A CUDA device will be faster.

Outputs:

- `model/convnext_tiny_best.pth` — float16 deployment checkpoint
- `artifacts/test_metrics.json` — deployed test metrics, confusion matrix, ROC, errors
- `artifacts/training_history.json` — loss/accuracy curves and run configuration
- `artifacts/test_predictions.npz` — test predictions and probabilities
- `artifacts/fp16_equivalence.json` — fp16 round-trip check
- `model/model_card.json` — model card and provenance
- `assets/test_mistakes/` — misclassified test examples

To regenerate only the notebook evidence and dataset samples:

```bash
python scripts/prepare_assets.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repository. Make sure
   `model/convnext_tiny_best.pth` is committed; the float16 file is under 100 MB.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Choose **New app**, select the repository and branch, and set the entry point
   to `app.py`.
4. Open **Advanced settings** and select **Python 3.12**. Streamlit Community
   Cloud does not use `runtime.txt` or `.python-version`; use the dropdown.
5. Deploy. Streamlit installs `requirements.txt` with CPU-only PyTorch wheels.
6. The default URL is `https://<app-name>.streamlit.app`.

No secrets, API keys or GPU are required. The checkpoint is committed with the
repository and is loaded once per server process. Free-tier apps may sleep after
inactivity and cold-start on the next visit.

## Reproducibility notes

- One split is used for every model; the split indices are not regenerated per model.
- Test metrics are computed after loading the float16 checkpoint, not from the
  training-time model.
- The original notebook is included at `notebook/IA2_Tomato_HuggingFace_Simple_Colab.ipynb`.
  Its saved outputs are the source of the original comparison metrics and curves.
- The deployed model is a retrain. Do not expect its metrics to exactly match the
  original run.

## Limitations

- 10 tomato leaf classes only; no other crops, fruit, stems or multi-label output.
- No disease severity estimate.
- Public-dataset performance does not guarantee field performance.
- Symptoms can overlap; image quality and background can change predictions.
- Grad-CAM shows model attention, not causal evidence.
- The tool does not recommend pesticides or treatment rates.

## Acknowledgements

Dataset: Project-AgML/tomato_leaf_disease. Paper: Zhang et al. (2024). Built with
PyTorch, torchvision, Hugging Face datasets, scikit-learn, Streamlit, Plotly and
fpdf2.
