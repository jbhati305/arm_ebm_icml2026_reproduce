# Claim 2: Soft Bellman equation and implicit lookahead


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_claim2_setup", "created_at": "2026-07-19T00:00:01+00:00", "title": "Setup, result, and scope"}
-->
## Target claim

Section 3.3 identifies the ARM–EBM mapping with the maximum-entropy soft
Bellman equation. The future log-partition `V_q(s⊕a)` is the implicit lookahead
value carried by the local ARM logit.

## Independent verification

On the full variable-length EOS tree, we sample arbitrary valid ARM logits,
apply the inverse mapping `r=M⁻¹(q)`, then solve the acyclic soft Bellman
recursion backward using `M(r)`. Every valid state-action entry is compared;
the forced invalid terminal actions are also checked to remain `-∞`.

## Result

**Reproduced.** The maximum fixed-point/round-trip residual is
**6.66×10⁻¹⁶** over all valid state-action pairs. The separate fixed-length
cross-check is **5.00×10⁻¹⁶**. Because the transition graph is a finite DAG,
this backward computation directly verifies the paper's acyclic, `γ=1`,
deterministic-transition special case of the soft Bellman equation.
