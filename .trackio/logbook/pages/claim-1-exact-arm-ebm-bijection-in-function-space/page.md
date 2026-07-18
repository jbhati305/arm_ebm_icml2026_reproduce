# Claim 1: Exact ARM–EBM bijection in function space


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_claim1_setup", "created_at": "2026-07-19T00:00:00+00:00", "title": "Setup, result, and scope"}
-->
## Target claim

Proposition 3.2 states that
the backward mapping `q(s,a)=r(s,a)` for `a=EOS`, and
`q(s,a)=r(s,a)+V_q(s⊕a)` otherwise, is a bijection and preserves the
complete-response distribution.

## Independent verification

We enumerate the complete variable-length response space with vocabulary
`V=8`, maximum response length `T=4`, and an explicit EOS action. This contains
585 valid responses, including EOS-only and all responses ending in EOS by the
maximum horizon. Invalid non-EOS actions at depth `T-1` receive `-∞`, exactly as
required by equation (3). Random immediate rewards are mapped backward into ARM
logits, and every EBM and ARM log probability is computed by exact summation.

## Result

**Reproduced.** The maximum absolute log-probability discrepancy across all 585
valid responses is **1.78×10⁻¹⁵**. An independent fixed-length `V=8, T=4`
cross-check over all 4,096 sequences gives **3.55×10⁻¹⁵**. Four regression tests
cover probability preservation and both fixed- and variable-length inverse
round trips.

This is a full function-space verification of the stated finite-domain
bijection, not a claim that a finite Transformer realizes the mapping exactly.


---
<!-- trackio-cell
{"type": "code", "id": "cell_ad59d796fb79", "created_at": "2026-07-18T20:26:07+00:00", "title": "Complete CPU reproduction command", "command": ["uv", "run", "--cache-dir", "/tmp/arm_ebm_uv_cache", "python", "-m", "repro_arm_ebm.run", "--steps", "3000", "--output-dir", "outputs/cpu_exact"], "exit_code": 0, "duration_s": 9.639}
-->
````bash
$ uv run --cache-dir /tmp/arm_ebm_uv_cache python -m repro_arm_ebm.run --steps 3000 --output-dir outputs/cpu_exact
````

exit 0 · 9.6s


````output
{
  "paper": "Autoregressive Language Models are Secretly Energy-Based Models",
  "scope": "Exact CPU verification in function space: variable-length EOS responses for Claims 1-2 and both fixed-horizon sequence spaces from Appendix C for Claim 3; not a finite-Transformer/LLM-scale replication.",
  "configuration": {
    "vocab_size": 8,
    "horizon": 4,
    "sequences": 4096,
    "steps": 3000,
    "seed": 7,
    "device": "CPU"
  },
  "claim_1_bijection": {
    "setting": "Variable-length responses with EOS and maximum length T",
    "valid_responses": 585,
    "max_abs_log_probability_difference": 1.7763568394002505e-15,
    "fixed_length_cross_check": 3.552713678800501e-15,
    "passed": true
  },
  "claim_2_soft_bellman": {
    "max_abs_fixed_point_residual": 6.661338147750939e-16,
    "fixed_length_cross_check": 4.996003610813204e-16,
    "passed": true
  },
  "claim_3_same_minima": {
    "target_entropy": 5.053137334770945,
    "analytic_ebm_gap": 0.0,
    "analytic_arm_gap": 0.0,
    "analytic_probability_error_v8_t4": 5.329070518200751e-15,
    "analytic_probability_error_v4_t8": 1.5987211554602254e-14,
    "trained_ebm_final_gap": 5.997902619014894e-09,
    "trained_arm_final_gap": 5.530588795821245e-07,
    "centred_arm_vs_mapped_ebm_linf": 0.007335004327232216,
    "training_seconds": 9.24572757799615
  },
  "claim_4_kl_bound": {
    "trials": 40,
    "directions_per_trial": 2,
    "max_kl_over_bound": 0.07152984733247113,
    "all_bounds_hold": true
  },
  "claim_5_complexity": {
    "complexity_table": [
      {
        "vocab_size": 8,
        "horizon": 1,
        "complete_sequences": 8,
        "context_action_logits": 8
      },
      {
        "vocab_size": 8,
        "horizon": 2,
        "complete_sequences": 64,
        "context_action_logits": 72
      },
      {
        "vocab_size": 8,
        "horizon": 3,
        "complete_sequences": 512,
        "context_action_logits": 584
      },
      {
        "vocab_size": 8,
        "horizon": 4,
        "complete_sequences": 4096,
        "context_action_logits": 4680
      },
      {
        "vocab_size": 8,
        "horizon": 5,
        "complete_sequences": 32768,
        "context_action_logits": 37448
      },
      {
        "vocab_size": 8,
        "horizon": 6,
        "complete_sequences": 262144,
        "context_action_logits": 299592
      }
    ],
    "interpretation": "Enumerating every complete sequence grows as V^T; the backwards EBM-to-ARM sweep must visit this exponentially growing tree."
  },
  "environment": {
    "python": "3.12.3",
    "numpy": "2.5.1",
    "platform": "Linux-6.17.0-35-generic-x86_64-with-glibc2.39"
  }
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_9da63491c467", "created_at": "2026-07-18T20:26:07+00:00", "title": "Artifact: kl_bound_trials.csv", "path": "outputs/cpu_exact/kl_bound_trials.csv", "size": 4344, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/cpu_exact/kl_bound_trials.csv` · dataset · 4.3 kB

https://huggingface.co/buckets/jbhati305/arm-ebm-icml2026-artifacts#logbook-files/outputs/cpu_exact/kl_bound_trials.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_2be481c62571", "created_at": "2026-07-18T20:26:07+00:00", "title": "Artifact: training_history.csv", "path": "outputs/cpu_exact/training_history.csv", "size": 3080, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/cpu_exact/training_history.csv` · dataset · 3.1 kB

https://huggingface.co/buckets/jbhati305/arm-ebm-icml2026-artifacts#logbook-files/outputs/cpu_exact/training_history.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_763509953144", "created_at": "2026-07-18T20:26:07+00:00", "title": "Artifact: complexity.csv", "path": "outputs/cpu_exact/complexity.csv", "size": 145, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/cpu_exact/complexity.csv` · dataset · 145 B

https://huggingface.co/buckets/jbhati305/arm-ebm-icml2026-artifacts#logbook-files/outputs/cpu_exact/complexity.csv
