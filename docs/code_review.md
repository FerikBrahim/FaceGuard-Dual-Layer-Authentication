# Code review and refactoring decisions

The repository was reconstructed from the uploaded paper, the uploaded `deepfakeEvaluation.ipynb`, and the FaceGuard watermarking/attack code available in the conversation.

## Major issues corrected

1. **Hard-coded Google Drive paths** were replaced by command-line arguments and configuration files.
2. **Notebook package installation cells** were moved into `requirements.txt`, `requirements-colab.txt`, and `environment.yml`.
3. **PyTorch 2.6 checkpoint loading** now uses restricted loading by default and an explicit trusted-checkpoint option for legacy files.
4. **Duplicate notebook cells and thresholds** were consolidated into reusable modules.
5. **Triplet-level random splitting** can leak identities between training and validation. The new dataset utilities provide identity-disjoint splitting.
6. **Online semi-hard mining terminology** did not match the uploaded training code, which uses preconstructed triplets and a standard cosine triplet loss. A genuine batch semi-hard implementation is included separately and is not enabled implicitly.
7. **Architecture naming** is standardized on InceptionResNetV1, matching the principal implementation and methodology. Earlier ResNet-50 wording should be removed from the paper.
8. **Framework naming** is standardized on PyTorch/facenet-pytorch. The paper's implementation section should not list TensorFlow unless a separate TensorFlow implementation is actually used.
9. **Synthetic manipulation terminology** is corrected: smoothing, color transfer, and affine warping are controlled synthetic manipulations, not trained deepfakes.
10. **Expected versus measured results** are separated. The pipeline never inserts proposed NCC or BER values into result tables.
11. **Watermark-key parsing** ignores non-key metadata fields while still requiring every actual extraction field.
12. **Metric-key mismatches** are removed by using one canonical result schema.

## Colab source status

The supplied Colab URLs required Google authentication in the available web view, so their private notebook contents could not be fetched directly. The uploaded `deepfakeEvaluation.ipynb` is preserved under `notebooks/archive/` and was reviewed locally. Export the remaining notebooks as `.ipynb` files and place them in `notebooks/archive/` for a line-by-line merge.
