# ARM–EBM ICML 2026 reproduction

Independent, CPU-only reproduction of the finite-state theoretical and
numerical settings in Blondel et al., *Autoregressive Language Models are
Secretly Energy-Based Models: Insights into the Lookahead Capabilities of
Next-Token Prediction*.

- [Published ICML 2026 reproduction logbook](https://huggingface.co/spaces/jbhati305/repro-autoregressive-language-models-are-secretly-energy-based-models-insights-into-the-lookahea)
- [Reproduction bundle and raw evidence](https://huggingface.co/buckets/jbhati305/arm-ebm-icml2026-artifacts#arm-ebm-icml2026-reproduction/reproduction-bundle:latest)
- [Hugging Face paper page](https://huggingface.co/papers/2512.15605)
- [OpenReview submission](https://openreview.net/forum?id=997oprE4sh)

The public Space is tagged `icml2026-repro` and `paper-997oprE4sh`. Its
canonical logbook contains an Index, Executive summary, one page for each of
the five anchored claims, and a Conclusion with the downloadable bundle.

This project tests the claims that can be exactly enumerated on CPU:

1. the Proposition 3.2 ARM↔EBM bijection;
2. the Section 3.3 soft Bellman fixed-point form;
3. the Proposition 4.1 equality of optimum NLLs in tabular function space;
4. the Proposition 59 KL upper bound; and
5. the exponential fixed-horizon sequence-space growth behind explicit EBM→ARM conversion.

Claims 1–2 use the paper's full variable-length EOS construction. Claim 3 uses
both fixed-horizon sequence spaces from Appendix C. The project does not present
these exact function-space checks as a Transformer or LLM-scale replication.

## Results

| Claim | Verification | Result |
| --- | --- | --- |
| 1 — exact ARM↔EBM bijection | Exhaustive EOS-aware enumeration of 585 responses, plus 4,096 fixed-length sequences | Maximum log-probability error `1.78e-15` |
| 2 — soft Bellman correspondence | Exact inverse mapping and backward fixed-point reconstruction | Maximum residual `6.66e-16` |
| 3 — identical NLL minima | Exact expected risks on `V=8,T=4` and `V=4,T=8`, plus 3,000 optimization steps | Analytical ARM and EBM gaps both `0.0` |
| 4 — KL error bound | 40 perturbation trials in both KL directions | `80/80` bounds held; maximum KL/bound ratio `0.07153` |
| 5 — conversion cost | Exact sequence-tree counts for `V=8`, `T=1…6` | Growth from 8 to 262,144 complete sequences |

All numerical checks use float64 and a fixed seed. The complete training-backed
run takes about 10 seconds on CPU. No GPU or Hugging Face Job is required for
the exhaustively enumerable settings tested here.

## Reproduce

Run the complete locked workflow:

```bash
./reproduce_cpu.sh
```

Or run each stage separately:

```bash
uv sync --cache-dir /tmp/arm_ebm_uv_cache
uv run --cache-dir /tmp/arm_ebm_uv_cache python -m pytest
uv run --cache-dir /tmp/arm_ebm_uv_cache python -m repro_arm_ebm.run
```

The workflow runs four regression tests and writes the machine-readable report,
raw CSV evidence, an HTML convergence plot, and the Posterly source/previews to
`outputs/cpu_exact/`.

## Repository layout

- `repro_arm_ebm/`: exact mappings, probability calculations, optimization,
  KL checks, and complexity experiment.
- `tests/`: fixed- and variable-length round-trip and probability tests.
- `outputs/cpu_exact/`: JSON report, raw CSVs, plots, and validated poster.
- `.trackio/logbook/`: canonical published challenge logbook source.
- `scripts/register_bundle.py`: Trackio reproduction-bundle registration.
- `REPRODUCTION_BUNDLE.md`: concise artifact inventory and rerun notes.

## Validation and publication

The local project passes:

- `4/4` pytest regression tests;
- the official ICML 2026 logbook validator;
- Posterly preflight, layout measurement, strict polish, and one-page A2 PDF
  verification.

The public artifact Bucket contains the complete reproduction bundle and the
three raw CSV evidence files linked from the claim pages. Authentication uses
the `HF_TOKEN` environment variable; `.env` is gitignored and credentials must
never be committed or copied into the logbook.
