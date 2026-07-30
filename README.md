# FaceGuard: Dual-Layer Face Authentication

Official research-code package for:

> **Dual-layer face authentication against spoofing and deepfakes using palmprint watermarking and Siamese verification**

FaceGuard authenticates both **who the subject is** and **whether the submitted
face image remains intact**. The framework combines a palmprint-derived binary
watermark embedded into the face image with an InceptionResNetV1 Siamese verifier.
A query is accepted only when both branches succeed:

$$
A_f = A_s \land A_w.
$$

| Siamese branch \(A_s\) | Watermark branch \(A_w\) | Outcome |
|---:|---:|---|
| 1 | 1 | Genuine |
| 1 | 0 | Watermark tampering |
| 0 | 1 | Identity spoofing or face-swap detected |
| 0 | 0 | Rejected |

## What is included

This update consolidates the uploaded Colab projects and categorizes them into:

- **Watermarking:** CompCode generation, vector DT-CWT–SVD-QIM, robust pairwise
  DT-CWT–SVD embedding, extraction, key sensitivity, imperceptibility, and
  conventional attack testing.
- **Attacks:** JPEG/noise/filtering/geometric attacks, face swapping, complete
  identity replacement, local facial tampering, and controlled synthetic
  manipulation.
- **Siamese networks:** InceptionResNetV1 triplet training, validation-derived
  threshold selection, checkpoint loading, and a frozen VGGFace2 zero-shot
  baseline.
- **Deepfake evaluation:** identity-aware real/fake datasets, threshold
  calibration, APCER/BPCER/ACER, per-method analysis, identity-disjoint folds,
  bootstrap confidence intervals, and McNemar paired comparison.
- **Fusion and reporting:** four-outcome authentication, CSV/JSON/Excel/LaTeX
  tables, contact sheets, and run metadata.

The categorized notebooks are documented in
[`docs/notebook_catalog.md`](docs/notebook_catalog.md).

## Methodology

### 1. Palmprint CompCode watermark

Six Gabor orientations \(0^\circ,30^\circ,60^\circ,90^\circ,120^\circ,150^\circ\)
produce a winner-take-all orientation map. Block histograms are quantized into a
deterministic 128-bit palmprint-derived watermark.

### 2. DT-CWT–SVD embedding

Face images are processed at \(512\times512\). The luminance channel is decomposed
with a two-level Dual-Tree Complex Wavelet Transform, and SVD is applied to
selected directional-subband blocks. Two maintained implementations are provided:

1. `PairwiseDTCWTSVDWatermarker`: blind pairwise singular-value ordering with
   spread masking and repeated embedding; used by the integrated pipeline.
2. `VectorWatermarkSystem`: notebook-derived blind SVD-QIM implementation with
   DT-CWT and optional DWT fallback, robustness testing, and key sensitivity.

Never mix results from these implementations without reporting the exact config.

### 3. Siamese verification

The primary network uses an InceptionResNetV1 encoder initialized from VGGFace2.
Faces are detected/aligned and resized to \(160\times160\times3\). A projection
head generates L2-normalized 512-dimensional embeddings. Cosine triplet loss uses
margin \(m=0.8\). Verification uses cosine similarity:

$$
s_s =
\cos\!\left(
f\!\left(R(I_r)\right),
f\!\left(R(I_q)\right)
\right),
\qquad
A_s =
\mathbb{1}\!\left[s_s \geq \theta_s\right].
$$


The threshold must be estimated from held-out validation identities. The code
supports both standard preconstructed triplets and genuine batch semi-hard
mining; these must be described separately in publications.

### 4. Watermark verification

The suspected query is transformed with the same key and parameters used during
embedding. The extracted 128-bit vector is compared with the enrolled vector by
NCC and BER. The strict defaults are `NCC >= 0.90` and `BER <= 0.05`.

NCC/BER values are exported only when the exact original vector and extraction
key are available. The repository does not fabricate missing watermark results.

## Repository structure

```text
faceguard-dual-layer-authentication/
├── configs/                      # Paper, watermark, Siamese, and deepfake configs
├── docs/                         # Methodology, notebook catalog, provenance, schemas
├── models/                       # Put face_verification_model.pth here
├── notebooks/
│   ├── colab/watermarking/       # Watermarking and attack notebooks
│   ├── colab/siamese/            # Training and zero-shot notebooks
│   ├── colab/evaluation/         # Deepfake/statistical notebook
│   └── archive/                  # Archive policy
├── outputs/                      # Generated metrics, images, and tables
├── sample_images/                # Authorized sample-image workspace
├── scripts/                      # Maintained CLI workflows
├── src/faceguard/                # Installable Python package
└── tests/                        # Automated unit tests
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/<YOUR-ACCOUNT>/faceguard-dual-layer-authentication.git
cd faceguard-dual-layer-authentication
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

For notebook reproduction:

```bash
pip install -e ".[notebooks]"
```

### Google Colab

```python
!git clone https://github.com/<YOUR-ACCOUNT>/faceguard-dual-layer-authentication.git
%cd faceguard-dual-layer-authentication
!pip install -r requirements-colab.txt
!pip install -e .
```

Restart the runtime after replacing binary scientific packages.

## Data organization

### Sample images

```text
sample_images/
├── palmprints/
├── faces/original/
├── faces/reference/
├── faces/donor/
├── faces/watermarked/
└── generated/
```

### Siamese triplets

```text
data/siamese/
├── anchor/<identity>/*.jpg
├── positive/<identity>/*.jpg
└── negative/<identity>/*.jpg
```

Training and validation are split by identity to prevent subject leakage.

### Deepfake dataset

```text
data/deepfake/
├── real/<identity>/*.jpg
└── fake/<method>/<identity>/*.jpg
```

Do not collapse all real and fake images into one pseudo-identity. APCER/BPCER
requires identity-aware claimed-identity trials.

## Principal workflows

### Generate the palmprint watermark

```bash
python scripts/generate_watermark.py \
  --palmprint sample_images/palmprints/palm_01.png \
  --output outputs/watermark_128.json
```

### Embed with the robust pairwise implementation

```bash
python scripts/embed_watermark.py \
  --face sample_images/faces/original/face_01.jpg \
  --watermark outputs/watermark_128.json \
  --output sample_images/faces/watermarked/face_01_watermarked.png \
  --key outputs/watermark_key.json
```

### Run the notebook-derived SVD-QIM implementation

```bash
python scripts/run_vector_watermark.py \
  --face sample_images/faces/original/face_01.jpg \
  --watermark outputs/watermark_128.json \
  --out_dir outputs/vector_qim \
  --transform dtcwt \
  --delta 4.0 \
  --seed 2026
```

### Batch-watermark labelled face images

```bash
python scripts/batch_watermark_faces.py \
  --input-root data/labelled_faces \
  --output-root outputs/watermarked_labelled_faces \
  --watermark outputs/watermark_128.json
```

### Train the Siamese model

```bash
python scripts/train_siamese.py \
  --anchor-root data/siamese/anchor \
  --positive-root data/siamese/positive \
  --negative-root data/siamese/negative \
  --checkpoint models/face_verification_model.pth \
  --epochs 15 --batch-size 16
```

### Evaluate the frozen zero-shot baseline

Create a CSV with `image_a,image_b,label`, then run:

```bash
python scripts/evaluate_zero_shot.py \
  --pairs-csv data/verification_pairs.csv \
  --output outputs/zero_shot_metrics.json
```

### Generate controlled attacks

```bash
python scripts/generate_attacks.py \
  --watermarked sample_images/faces/watermarked/face_01_watermarked.png \
  --donor sample_images/faces/donor/donor_01.jpg
```

### Evaluate full dual-layer authentication

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

Omit both `--watermark` and `--key` for an explicitly labeled Siamese-only table.

### Evaluate real deepfake data

```bash
python scripts/evaluate_deepfake_dataset.py \
  --dataset-root data/deepfake \
  --checkpoint models/face_verification_model.pth \
  --threshold <VALIDATION_THRESHOLD> \
  --trusted-checkpoint \
  --calibrate-domain \
  --output outputs/deepfake_evaluation.json
```

### Compare zero-shot and fine-tuned predictions

```bash
python scripts/compare_model_predictions.py \
  --csv outputs/paired_predictions.csv \
  --threshold-a <ZERO_SHOT_THRESHOLD> \
  --threshold-b <FINE_TUNED_THRESHOLD>
```

## Checkpoint handling

Place the model at `models/face_verification_model.pth`, use Git LFS, attach it
to a GitHub Release, or download it from a public Google Drive link:

```bash
python scripts/download_checkpoint.py "<GOOGLE-DRIVE-SHARE-URL>"
```

PyTorch 2.6 may require `--trusted-checkpoint` for legacy checkpoints containing
NumPy metadata. Enable it only for a checkpoint created by you or received from
a fully trusted source.

## Manuscript reference values

The manuscript reports Siamese accuracy 96.18%, AUC 0.9909, EER 4.04%, average
PSNR 65.5357 dB, SSIM 0.999910, and MSE 0.021604. These values are reference
results, not hard-coded outputs. Reproduction requires the same identities,
splits, images, watermark vectors, exact keys, checkpoint, and attack parameters.

## Reproducibility and research integrity

- Store the exact original watermark and JSON key with every watermarked image.
- Calibrate thresholds on identities disjoint from the final test set.
- Label controlled smoothing/warping as synthetic manipulation, not a trained deepfake.
- Do not report expected NCC/BER as measured values.
- Report the number of identities, genuine trials, attack trials, and attack methods.
- Use authorized biometric images and do not commit private datasets.

See [`docs/reproducibility_workflows.md`](docs/reproducibility_workflows.md) and
[`docs/paper_review.md`](docs/paper_review.md).

## Tests

```bash
pytest -q
```

## Citation

Use `CITATION.cff`. Update publication venue, DOI, and release version after
acceptance.

## License

MIT for repository code. Dataset and pretrained-model licenses remain separate.
