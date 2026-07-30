# Security and biometric-data handling

## Checkpoint safety

PyTorch checkpoints may contain pickle data. Use the `--trusted-checkpoint` option only for a model created by the research team or obtained from a fully trusted source. Never unrestrictedly load an unknown `.pth` or `.pt` file.

## Biometric privacy

Face and palmprint images are sensitive biometric data. The repository intentionally contains no participant data. Store research samples outside public Git history unless redistribution is explicitly authorized. Remove metadata, use subject codes rather than names, and follow the consent, ethics, and dataset-license conditions governing the experiment.

## Watermark keys

The original 128-bit watermark and extraction key are authentication secrets. Do not publish production keys. For reproducible research, use synthetic or consented samples and clearly label demonstration keys.

## Reporting vulnerabilities

For security-sensitive implementation issues, contact the corresponding research team privately before opening a public issue. Include the affected module, a minimal reproduction, and the potential impact without sharing biometric records or private keys.
