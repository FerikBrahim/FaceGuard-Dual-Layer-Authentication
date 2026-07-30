# Methodology mapping

| Paper component | Repository implementation |
|---|---|
| Palmprint CompCode | `src/faceguard/watermark/compcode.py` |
| 128-bit watermark | `CompCodeGenerator(output_bits=128)` |
| DT-CWT–SVD embedding | `PairwiseDTCWTSVDWatermarker.embed()` |
| Semi-blind extraction | `PairwiseDTCWTSVDWatermarker.extract()` with saved JSON key |
| InceptionResNetV1 Siamese encoder | `src/faceguard/siamese/model.py` |
| Cosine triplet loss | `CosineTripletLoss` |
| Optional true semi-hard mining | `BatchSemiHardTripletLoss` |
| Cosine verification | `SiameseVerifier.verify()` |
| Controlled attacks | `src/faceguard/attacks/` |
| Decision-level fusion | `src/faceguard/fusion/authentication.py` |
| Full evaluation | `src/faceguard/pipeline.py` |

## Resolutions

- Watermark generation input: palmprint normalized to 256×256.
- Watermark embedding and extraction: face processed at 512×512.
- Siamese encoder input: independently detected/aligned face at 160×160×3.

## Measurement policy

NCC and BER are valid only when the exact original 128-bit watermark and exact extraction key are available. The key must be generated during embedding and stored with experiment outputs. A synthetic key may test code execution but cannot support a paper result.
