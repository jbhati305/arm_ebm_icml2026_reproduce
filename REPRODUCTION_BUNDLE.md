# CPU reproduction bundle

This repository is the reproduction bundle for arXiv:2512.15605.

It contains:

- `repro_arm_ebm/`: exact tabular ARM/EBM mapping and experiment runner;
- `tests/`: four exact mapping and probability regression tests;
- `outputs/cpu_exact/`: JSON report, raw CSVs, training plot, and validated poster;
- `.trackio/logbook/`: canonical ICML 2026 challenge logbook pages; and
- `pyproject.toml` and `uv.lock`: pinned environment metadata.
- `scripts/register_bundle.py`: explicit Trackio artifact registration.

Rerun the complete CPU reproduction with:

```bash
./reproduce_cpu.sh
```

Claims 1–2 use the variable-length EOS construction. Claim 3 checks both
fixed-horizon sequence spaces from Appendix C (V=8, T=4 and V=4, T=8). All
spaces are fully enumerable on CPU. This bundle does not reproduce the paper's
finite Transformer parameterizations or make LLM-scale claims.
