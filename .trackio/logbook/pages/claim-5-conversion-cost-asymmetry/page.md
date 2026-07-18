# Claim 5: Conversion-cost asymmetry


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_claim5_setup", "created_at": "2026-07-19T00:00:04+00:00", "title": "Setup, result, and scope"}
-->
## Target claim

Explicit EBM→ARM conversion requires a backward sweep over an exponentially
large sequence tree, while evaluating the inverse ARM→EBM correction along a
specified sequence costs `O(VT)`.

## Independent verification

We count every sequence and context-action logit for `V=8` as the horizon grows
from 1 to 6. Complete sequences grow as `V^T`; the explicit backward map must
visit the corresponding prefix tree. The inverse formula subtracts one
next-state log-partition per edge of a given path; each partition sums over
`V` actions, giving `O(VT)` work across `T` positions.

## Result

**Mechanism reproduced.** Complete-sequence counts are 8, 64, 512, 4,096,
32,768, and 262,144 for `T=1…6`; full-tree context-action counts are 8, 72,
584, 4,680, 37,448, and 299,592. This directly exhibits the exponential
enumeration barrier. We report this as a complexity/mechanism check, not a
large-scale wall-clock benchmark.


---
<!-- trackio-cell
{"type": "figure", "id": "cell_d26d9d7cd3c6", "created_at": "2026-07-18T20:27:06+00:00", "title": "Exact sequence-tree growth"}
-->
````html
<p>Exact path and context-action counts for V=8.</p>
````

````raw
vocab_size,horizon,complete_sequences,context_action_logits
8,1,8,8
8,2,64,72
8,3,512,584
8,4,4096,4680
8,5,32768,37448
8,6,262144,299592

````
