# Reproducibility workflows

## Workflow A — watermarking only

1. Generate a true 128-bit CompCode watermark from a palmprint.
2. Embed it into a 512×512 face image.
3. Save both the watermarked image and exact JSON extraction key.
4. Extract from the clean image and report NCC/BER.
5. Apply the conventional attack suite and repeat extraction.

Use either the paper-oriented pairwise implementation or the notebook-derived
SVD-QIM implementation, but state which method and configuration generated each
table.

## Workflow B — Siamese training

1. Organize anchor, positive watermarked, and negative identity images by identity.
2. Split identities—not triplets—between training and validation.
3. Fine-tune InceptionResNetV1 for 15 epochs using cosine triplet loss.
4. Determine `optimal_threshold` and `eer_threshold` from validation scores.
5. Save the threshold inside the checkpoint and as `face_metrics.json`.

## Workflow C — zero-shot comparison

Evaluate the frozen VGGFace2 InceptionResNetV1 model on the exact validation
pairs used for the fine-tuned model. Export paired predictions so McNemar's test
can compare the models without confounding the test set.

## Workflow D — controlled facial manipulation

Use one fixed watermarked target and one donor for the proof-of-concept table.
Report measured Siamese values. Report NCC, BER, `A_w`, and `A_f` only when the
original watermark and exact extraction key are available.

## Workflow E — real deepfake datasets

Use `real/<identity>` and `fake/<method>/<identity>` organization. Keep threshold
calibration identities separate from test identities. Report APCER, BPCER, ACER,
attack AUC, attack EER, per-method APCER, and the number of identities/trials.
