# Colab notebook catalog

The uploaded code is organized by experimental role. All repository copies have
execution outputs removed while preserving code cells. Original names and hashes
are recorded in `notebook_provenance.json`.

| Repository notebook | Category | Role | Manuscript status |
|---|---|---|---|
| `watermarking/01_vector_watermark_baseline.ipynb` | Watermarking | Blind vector DT-CWT–SVD-QIM embedding, extraction, imperceptibility, robustness, and key sensitivity | Maintained alternative implementation |
| `watermarking/02_vector_watermark_extended_attacks.ipynb` | Watermarking + attacks + fusion | Batch experiments, conventional attacks, facial manipulations, and measured table export | Combined experimental notebook |
| `watermarking/03_batch_watermarked_faces_legacy.ipynb` | Legacy experiment | DWT watermarking and DINOv2-based Siamese experiment on labelled faces | Archived; not the paper-default InceptionResNetV1/DT-CWT path |
| `siamese/01_inceptionresnet_triplet_training.ipynb` | Siamese network | InceptionResNetV1 fine-tuning, triplet training, ROC/EER evaluation, and checkpoint generation | Primary network workflow |
| `siamese/02_vggface2_zero_shot_baseline.ipynb` | Baseline | Frozen VGGFace2 model evaluated on the same verification pairs | Baseline workflow |
| `evaluation/01_deepfake_statistical_evaluation.ipynb` | Evaluation | Identity-aware deepfake testing, threshold calibration, APCER/BPCER, k-fold analysis, bootstrap confidence intervals, and McNemar comparison | Extended evaluation workflow |

## Canonical package mapping

| Notebook concept | Maintained source |
|---|---|
| Vector SVD-QIM watermarking | `src/faceguard/watermark/vector_system.py` |
| Pairwise robust DT-CWT–SVD watermarking | `src/faceguard/watermark/dtcwt_svd.py` |
| CompCode generation | `src/faceguard/watermark/compcode.py` |
| Siamese model and preprocessing | `src/faceguard/siamese/model.py` |
| Triplet loss | `src/faceguard/siamese/loss.py` |
| Zero-shot baseline | `src/faceguard/siamese/zero_shot.py` |
| Statistical tests | `src/faceguard/siamese/statistics.py` |
| Deepfake evaluation | `src/faceguard/siamese/deepfake.py` |
| Controlled facial attacks | `src/faceguard/attacks/facial_manipulation.py` |
| Final authentication fusion | `src/faceguard/fusion/authentication.py` |

## Important interpretation

The DINOv2/DWT labelled-face notebook is retained for provenance and possible
ablation work. The research paper's principal architecture is InceptionResNetV1
with palmprint-derived DT-CWT–SVD watermarking. Results from alternative
notebooks should not be merged into the paper tables without explicit labeling.
