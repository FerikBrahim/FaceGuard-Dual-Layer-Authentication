# FaceGuard: Dual-Layer Face Authentication

Research implementation accompanying the paper:

> **Dual-layer face authentication against spoofing and deepfakes using palmprint watermarking and Siamese verification**

FaceGuard combines two complementary security mechanisms:

1. **Palmprint-based watermark verification**, which evaluates the integrity and provenance of a face image.
2. **Siamese face verification**, which evaluates whether the submitted face matches the enrolled identity.

A query is accepted only when both branches succeed:

$$
A_f = A_s \land A_w.
$$

Here, $A_s$ is the Siamese decision, $A_w$ is the watermark-verification decision, and $A_f$ is the final fused decision.

| Siamese decision $A_s$ | Watermark decision $A_w$ | Authentication outcome |
|---:|---:|---|
| 1 | 1 | Genuine |
| 1 | 0 | Watermark tampering |
| 0 | 1 | Identity spoofing or face-swap detected |
| 0 | 0 | Rejected |

---

## 1. Repository scope

This repository consolidates and refactors the Colab implementations developed for the research paper. The notebooks and Python modules are organized into the following categories:

- **Palmprint watermark generation**
  - Competitive Code extraction using six Gabor orientations.
  - Deterministic 128-bit watermark construction.

- **DT-CWT–SVD watermarking**
  - Watermark embedding in the luminance component of face images.
  - Semi-blind watermark extraction.
  - Normalized cross-correlation, bit-error rate, extraction accuracy, PSNR, SSIM, and MSE.

- **Watermark robustness attacks**
  - JPEG compression.
  - Gaussian, salt-and-pepper, and speckle noise.
  - Median, Gaussian, and average filtering.
  - Sharpening, histogram equalization, and contrast adjustment.
  - Resizing, rotation, cropping, and occlusion.

- **Facial-manipulation attacks**
  - Controlled face swapping.
  - Complete identity replacement.
  - Mouth-region blurring.
  - Eye-region occlusion.
  - Cheek-region alteration.
  - Controlled synthetic facial manipulation.

- **Siamese face verification**
  - InceptionResNetV1 initialized from VGGFace2.
  - Triplet-based metric learning.
  - Cosine-similarity verification.
  - Validation-based threshold calibration.
  - Frozen VGGFace2 zero-shot baseline.

- **Deepfake and attack evaluation**
  - Attack Presentation Classification Error Rate.
  - Bona Fide Presentation Classification Error Rate.
  - Average Classification Error Rate.
  - Receiver Operating Characteristic analysis, Area Under the Curve, and Equal Error Rate.
  - Bootstrap confidence intervals and paired McNemar comparisons.

- **Decision-level fusion and reporting**
  - Four authentication outcomes.
  - CSV, JSON, Excel, image, and LaTeX exports.

The categorized Colab notebooks are documented in [`docs/notebook_catalog.md`](docs/notebook_catalog.md).

---

## 2. Proposed methodology

### 2.1 Palmprint CompCode watermark generation

A palmprint image is processed using a bank of Gabor filters with six orientations:

$$
\theta \in \{0^\circ,30^\circ,60^\circ,90^\circ,120^\circ,150^\circ\}.
$$

At each pixel, the orientation producing the strongest response is retained through a winner-take-all rule. The resulting orientation map is divided into spatial blocks, and block histograms are quantized and concatenated to form a fixed-length 128-bit biometric watermark.

### 2.2 DT-CWT–SVD watermark embedding

Face images are processed at $512\times512$ pixels. The RGB image is converted to YCbCr, and watermark embedding is performed in the luminance channel. A two-level Dual-Tree Complex Wavelet Transform provides directionally selective subbands, while Singular Value Decomposition is applied to selected blocks.

The repository contains two maintained watermarking implementations:

1. **`PairwiseDTCWTSVDWatermarker`**  
   Uses pairwise singular-value ordering, repeated embedding, and a pseudo-random spreading mask. It is used by the integrated dual-layer pipeline.

2. **`VectorWatermarkSystem`**  
   Refactors the uploaded Colab implementation based on DT-CWT–SVD quantization, robustness evaluation, and key-sensitivity analysis.

Results from the two implementations must not be mixed unless the exact configuration, embedding rule, key, and threshold are reported.

### 2.3 Siamese face verification

The primary face encoder is InceptionResNetV1 initialized from VGGFace2. Detected faces are aligned and resized to $160\times160\times3$. The network produces L2-normalized 512-dimensional embeddings.

For an enrolled reference image $I_r$ and a query image $I_q$, cosine similarity is computed as:

$$
s_s = \cos\!\left(f\!\left(R(I_r)\right),f\!\left(R(I_q)\right)\right).
$$

The Siamese verification decision is:

$$
A_s =
\begin{cases}
1, & s_s \geq \theta_s,\\
0, & s_s < \theta_s.
\end{cases}
$$

Here, $R(\cdot)$ denotes face detection, alignment, and resizing; $f(\cdot)$ denotes the shared encoder; and $\theta_s$ is obtained from a held-out validation set.

### 2.4 Watermark extraction and verification

The suspected query image is processed using the same transform settings and extraction key used during embedding. The recovered 128-bit vector $\widehat{W}$ is compared with the enrolled watermark $W$ using normalized cross-correlation and bit-error rate.

The watermark decision is:

$$
A_w =
\begin{cases}
1, & \operatorname{NCC}(W,\widehat{W}) \geq \theta_w,\\
0, & \operatorname{NCC}(W,\widehat{W}) < \theta_w.
\end{cases}
$$

The default strict evaluation settings are:

```text
NCC threshold: 0.90
BER threshold: 0.05
Watermark length: 128 bits
```

Scientifically valid NCC and BER values require the exact original watermark and the exact extraction key generated during embedding. The repository does not replace missing measurements with expected values.

### 2.5 Decision-level fusion

The final decision is obtained by strict logical conjunction:

$$
A_f = A_s \land A_w.
$$

This rule ensures that a query is accepted only when the facial identity is verified and the embedded palmprint watermark remains valid.

---

## 3. Repository structure

```text
faceguard-dual-layer-authentication/
├── .github/workflows/              # Continuous-integration tests
├── configs/                        # Experiment configurations
├── docs/                           # Methodology, review, and reproducibility notes
├── models/                         # Trained Siamese checkpoint location
├── notebooks/
│   ├── colab/watermarking/         # Watermarking and robustness notebooks
│   ├── colab/siamese/              # Siamese training and zero-shot notebooks
│   ├── colab/evaluation/           # Deepfake and statistical evaluation
│   └── archive/                    # Archived notebook information
├── outputs/                        # Generated metrics, tables, keys, and images
├── sample_images/
│   ├── palmprints/                 # Authorized palmprint samples
│   ├── faces/original/             # Original host face images
│   ├── faces/reference/            # Enrolled reference faces
│   ├── faces/donor/                # Donor identities for controlled attacks
│   ├── faces/watermarked/          # Watermarked face images
│   └── generated/                  # Generated attack images
├── scripts/                        # Command-line experiment workflows
├── src/faceguard/                  # Installable Python package
└── tests/                          # Automated unit tests
```

---

## 4. Trained model checkpoint

The trained checkpoint is **not included in this repository archive**. After training the Siamese network, place the checkpoint at exactly:

```text
faceguard-dual-layer-authentication/models/face_verification_model.pth
```

The relative path used by the scripts is:

```text
models/face_verification_model.pth
```

### Download the trained checkpoint

The trained Siamese checkpoint is hosted on Google Drive because it is too large for a standard GitHub browser upload:

[Download `face_verification_model.pth` from Google Drive](https://drive.google.com/file/d/1WwfZ-s6iovePSyCcL-ubx_gPLSdP61Bf/view?usp=sharing)

After downloading the file, place it in the repository at:

```text
models/face_verification_model.pth
```

The checkpoint can also be downloaded automatically with `gdown`:

```bash
python -m pip install gdown
mkdir -p models
gdown --fuzzy "https://drive.google.com/file/d/1WwfZ-s6iovePSyCcL-ubx_gPLSdP61Bf/view?usp=sharing" \
  -O models/face_verification_model.pth
```

In Google Colab, run:

```python
!pip -q install gdown
!mkdir -p models
!gdown --fuzzy "https://drive.google.com/file/d/1WwfZ-s6iovePSyCcL-ubx_gPLSdP61Bf/view?usp=sharing" \
  -O models/face_verification_model.pth
```

The Google Drive sharing permission must allow anyone with the link to view the file. Verify the download before running evaluation scripts.

### Saving the checkpoint during training

```python
from pathlib import Path
import torch

checkpoint_path = Path("models/face_verification_model.pth")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_metrics": final_metrics,
        "optimal_threshold": float(optimal_threshold),
    },
    checkpoint_path,
)

print(f"Checkpoint saved to: {checkpoint_path.resolve()}")
```

### Google Colab location

When the repository is cloned into `/content`, the corresponding Colab path is:

```text
/content/faceguard-dual-layer-authentication/models/face_verification_model.pth
```

When the model is stored permanently in Google Drive, a recommended path is:

```text
/content/drive/MyDrive/FaceGuard/models/face_verification_model.pth
```

Example:

```python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

CHECKPOINT_PATH = Path(
    "/content/drive/MyDrive/FaceGuard/models/face_verification_model.pth"
)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
```

For large checkpoints, use Git Large File Storage, a GitHub Release, or a public Google Drive link rather than committing the file directly to the normal Git history.

> **Security note:** PyTorch checkpoints use pickle-based serialization. Load a checkpoint with unrestricted deserialization only when it was created by you or obtained from a fully trusted source.

---

## 5. Installation

Python 3.10 or 3.11 is recommended.

### Linux and macOS

```bash
git clone https://github.com/<YOUR-ACCOUNT>/faceguard-dual-layer-authentication.git
cd faceguard-dual-layer-authentication

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

### Windows PowerShell

```powershell
git clone https://github.com/<YOUR-ACCOUNT>/faceguard-dual-layer-authentication.git
cd faceguard-dual-layer-authentication

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
```

### Notebook dependencies

```bash
pip install -e ".[notebooks]"
```

---

## 6. Google Colab setup

```python
!git clone https://github.com/<YOUR-ACCOUNT>/faceguard-dual-layer-authentication.git
%cd faceguard-dual-layer-authentication
!pip install -r requirements-colab.txt
!pip install -e .
```

Restart the Colab runtime after changing NumPy, pandas, SciPy, scikit-learn, or OpenCV versions. Binary scientific packages should be installed as a compatible set before they are imported.

---

## 7. Data organization

### 7.1 Sample images

```text
sample_images/
├── palmprints/
├── faces/original/
├── faces/reference/
├── faces/donor/
├── faces/watermarked/
└── generated/
```

Only use biometric images for which you have authorization. Private biometric datasets should not be committed to a public repository.

### 7.2 Siamese training data

```text
data/siamese/
├── anchor/<identity>/*.jpg
├── positive/<identity>/*.jpg
└── negative/<identity>/*.jpg
```

Training, validation, and testing should be split by identity to prevent subject leakage.

### 7.3 Deepfake evaluation data

```text
data/deepfake/
├── real/<identity>/*.jpg
└── fake/<method>/<identity>/*.jpg
```

Each fake image should retain its claimed identity and manipulation method. Do not collapse all real or fake images into a single pseudo-identity.

---

## 8. Main workflows

### 8.1 Generate a 128-bit palmprint watermark

```bash
python scripts/generate_watermark.py \
  --palmprint sample_images/palmprints/palm_01.png \
  --output outputs/watermark_128.json
```

### 8.2 Embed the watermark

```bash
python scripts/embed_watermark.py \
  --face sample_images/faces/original/face_01.jpg \
  --watermark outputs/watermark_128.json \
  --output sample_images/faces/watermarked/face_01_watermarked.png \
  --key outputs/watermark_key.json
```

The generated watermark key must be retained because it identifies the transform parameters and coefficient locations required during extraction.

### 8.3 Extract and verify the watermark

```bash
python scripts/extract_watermark.py \
  --image sample_images/faces/watermarked/face_01_watermarked.png \
  --watermark outputs/watermark_128.json \
  --key outputs/watermark_key.json \
  --output outputs/extraction_results.json
```

### 8.4 Run the notebook-derived vector watermarking implementation

```bash
python scripts/run_vector_watermark.py \
  --face sample_images/faces/original/face_01.jpg \
  --watermark outputs/watermark_128.json \
  --out_dir outputs/vector_qim \
  --transform dtcwt \
  --delta 4.0 \
  --seed 2026
```

### 8.5 Batch-watermark labelled face images

```bash
python scripts/batch_watermark_faces.py \
  --input-root data/labelled_faces \
  --output-root outputs/watermarked_labelled_faces \
  --watermark outputs/watermark_128.json
```

### 8.6 Train the Siamese model

```bash
python scripts/train_siamese.py \
  --anchor-root data/siamese/anchor \
  --positive-root data/siamese/positive \
  --negative-root data/siamese/negative \
  --checkpoint models/face_verification_model.pth \
  --epochs 15 \
  --batch-size 16
```

### 8.7 Evaluate the frozen VGGFace2 baseline

Create a CSV file containing `image_a,image_b,label`, then run:

```bash
python scripts/evaluate_zero_shot.py \
  --pairs-csv data/verification_pairs.csv \
  --output outputs/zero_shot_metrics.json
```

### 8.8 Generate controlled facial attacks

```bash
python scripts/generate_attacks.py \
  --watermarked sample_images/faces/watermarked/face_01_watermarked.png \
  --donor sample_images/faces/donor/donor_01.jpg
```

### 8.9 Evaluate dual-layer authentication

```bash
python scripts/evaluate_dual_layer.py \
  --reference sample_images/faces/reference/face_01_reference.jpg \
  --watermarked sample_images/faces/watermarked/face_01_watermarked.png \
  --donor sample_images/faces/donor/donor_01.jpg \
  --checkpoint models/face_verification_model.pth \
  --siamese-threshold <VALIDATION_THRESHOLD> \
  --watermark outputs/watermark_128.json \
  --key outputs/watermark_key.json \
  --trusted-checkpoint
```

Use `--trusted-checkpoint` only for a model file created by you or obtained from a trusted source.

When the exact watermark and key are unavailable, omit both `--watermark` and `--key`. The resulting output must be identified as **Siamese-only evaluation**, not dual-layer authentication.

### 8.10 Evaluate a real deepfake dataset

```bash
python scripts/evaluate_deepfake_dataset.py \
  --dataset-root data/deepfake \
  --checkpoint models/face_verification_model.pth \
  --threshold <VALIDATION_THRESHOLD> \
  --trusted-checkpoint \
  --calibrate-domain \
  --output outputs/deepfake_evaluation.json
```

### 8.11 Compare zero-shot and fine-tuned predictions

```bash
python scripts/compare_model_predictions.py \
  --csv outputs/paired_predictions.csv \
  --threshold-a <ZERO_SHOT_THRESHOLD> \
  --threshold-b <FINE_TUNED_THRESHOLD>
```

---

## 9. Expected outputs

Depending on the workflow, the project may generate:

```text
outputs/
├── watermark_128.json
├── watermark_key.json
├── extraction_results.json
├── imperceptibility_results.csv
├── robustness_results.csv
├── siamese_metrics.json
├── face_metrics.json
├── facial_manipulation_results.csv
├── dual_layer_results.csv
├── security_summary.json
├── generated_figures/
└── latex_tables/
```

The exact filenames may vary between the refactored scripts and archived Colab notebooks. Every reported result should preserve its configuration, random seed, threshold, checkpoint identity, watermark vector, and extraction key.

---

## 10. Reference results reported in the manuscript

The manuscript reports the following reference values:

| Component | Metric | Reported value |
|---|---|---:|
| Siamese verification | Accuracy | 96.18% |
| Siamese verification | Area Under the Curve | 0.9909 |
| Siamese verification | Equal Error Rate | 4.04% |
| Watermark imperceptibility | Average PSNR | 65.5357 dB |
| Watermark imperceptibility | Average SSIM | 0.999910 |
| Watermark imperceptibility | Average MSE | 0.021604 |

These values are documentation targets, not hard-coded outputs. Reproduction requires the same dataset identities, splits, images, model checkpoint, threshold, watermark vector, extraction key, and attack parameters.

---

## 11. Reproducibility and research integrity

- Preserve the exact original watermark and extraction key for every watermarked image.
- Calibrate the Siamese threshold on held-out identities that are disjoint from the final evaluation set.
- Report whether triplets were preconstructed or selected using genuine online semi-hard mining.
- Describe the controlled smoothing and warping experiment as **controlled synthetic facial manipulation**, not as a deepfake generated by a trained synthesis model.
- Do not report expected NCC or BER values as measured experimental results.
- Do not claim zero false acceptance unless it is supported by the complete evaluated attack set.
- Report the number of identities, genuine trials, impostor trials, attack samples, and manipulation methods.
- Keep all biometric images authorized, anonymized where possible, and excluded from public version control unless redistribution is permitted.

Additional guidance is available in:

- [`docs/reproducibility_workflows.md`](docs/reproducibility_workflows.md)
- [`docs/paper_review.md`](docs/paper_review.md)
- [`docs/results_schema.md`](docs/results_schema.md)

---

## 12. Tests

Run the automated tests with:

```bash
pytest -q
```

---

## 13. Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Update the journal, year, volume, pages, Digital Object Identifier, and repository release after publication.

---

## 14. License

The repository code is distributed under the MIT License. Dataset licenses, biometric-image permissions, and pretrained-model licenses remain separate and must be respected independently.
