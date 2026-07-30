# Results schema

The dual-layer evaluation CSV contains:

- `Query condition`
- `Attack category`
- `Cosine similarity s_s`
- `A_s`
- `PSNR`
- `SSIM`
- `MSE`
- `NCC`
- `BER (%)`
- `A_w`
- `A_f`
- `Final outcome`

When the watermark vector or key is missing, NCC, BER, `A_w`, and `A_f` are left empty. The output must then be described as Siamese-only evaluation.

## Outcome rule

| `A_s` | `A_w` | Outcome |
|---:|---:|---|
| 1 | 1 | Genuine |
| 1 | 0 | Watermark tampering |
| 0 | 1 | Identity spoofing or face-swap detected |
| 0 | 0 | Rejected |
