# Contributing

Contributions should preserve the distinction between measured and expected results. New experiments must include configuration files, random seeds, dataset-split descriptions, and tests. Do not commit biometric images, model checkpoints, watermark keys, or dataset files without explicit permission.

Before submitting a pull request:

```bash
pytest -q
ruff check src scripts tests
```
