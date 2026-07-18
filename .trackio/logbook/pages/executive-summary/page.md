# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_exec_summary", "created_at": "2026-07-19T00:00:05+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-19T00:00:05+00:00"}
-->
The first three core function-space claims reproduce exactly on CPU: the
variable-length EOS-aware ARM↔EBM bijection agrees to `1.78×10⁻¹⁵`, its soft
Bellman round trip agrees to `6.66×10⁻¹⁶`, and the ARM/EBM analytical NLL gaps
are both zero on both sequence spaces used by the paper. The KL inequality also
holds in all 80 tested directions, while the conversion-cost experiment exposes
the predicted exponential tree growth. This is an exact tabular reproduction of
the paper's finite-domain theory plus a scaled numerical validation; it does not
reproduce finite Transformer training curves. The complete training-backed run
takes about 10 seconds on CPU at no incremental compute cost.

## Scope & cost

|  | This reproduction | Full replication |
| --- | --- | --- |
| Scope | All five anchored claims in tabular function space; both paper sequence spaces | All paper Transformer architectures, distributions, learning rates, and appendix plots |
| Hardware | CPU | Accelerator hardware was not specified in Appendix C |
| Compute time | ~10 s for training-backed run; analytical checks <1 s | 5,000 steps per model/configuration, multiple sweeps |
| Cost | $0 incremental cost | Not reported by the paper |
| Outcome | Claims 1–3 exact; Claim 4 supported in 80/80 checks; Claim 5 mechanism confirmed | Not attempted |



---
<!-- trackio-cell
{"type": "figure", "id": "cell_44c94a8aca03", "created_at": "2026-07-18T20:52:42+00:00", "title": "Final reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-18T20:52:43+00:00"}
-->
````html
<!DOCTYPE html>
<!-- poster_embed.html: generated from Chenruishuo/posterly's passing offline scaffold. -->
<!--
  ============================================================
  posterly-based reproduction poster
  PURPOSE: summarize the exact CPU evidence in the ICML 2026 logbook.
  CANVAS:  A2 portrait (420 × 594 mm), single card, no MathJax,
           no external images — renders fully offline, single
           page, fills the canvas.

  The example is a single card FILLED with real content -- not
  stretched with `flex: 1` + blank space to fake a full page (see
  SKILL.md "Gate C -- the same trap, one card": a half-empty
  stretched card passes every gate yet reads as a failed poster).
  It stays single-card so it's robust to font / Chromium revision
  drift; multi-card alignment is the templates/ gallery's job, not
  this install-verification fixture's.

  Use:
    cd examples/hello_world
    python ../../tools/poster_check.py preflight   poster.html
    python ../../tools/poster_check.py measure     poster.html
    python ../../tools/poster_check.py polish      poster.html --strict
    python ../../tools/render_preview.py           poster.html
    python ../../tools/poster_check.py verify-final poster_preview.pdf --from-html poster.html
  Expect five lines confirming success.
  ============================================================
-->
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ARM–EBM CPU reproduction</title>
<style>
  @page { size: A2 portrait; margin: 0; }

  :root {
    --accent:        #2D5F8B;
    --accent-deep:   #1F4566;
    --accent-light:  #E8F1F8;
    --gold:          #C9A24A;
    --text-primary:  #1A1A1A;
    --text-muted:    #777;
    --bg-page:       #F6F3F0;
    --bg-card:       #FFFFFF;
    --border-soft:   #D8D8D8;
    --u: 1.6px;
    --font-sans: "Inter","Helvetica Neue",sans-serif;
    --font-serif: "Charter","Source Serif Pro","Georgia",serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }

  /* BASE DEFENSES (see templates/*_neutral.html for the annotated
     version): orphan/widow wrap protection on prose, balance on the
     centered title. Keep this block in any skeleton, incl. custom. */
  p, li, dd, figcaption,
  .body-text, .caption, .callout, .section-title { text-wrap: pretty; }
  .title { text-wrap: balance; }
  html, body { background: #2b2b2b; font-family: var(--font-serif); color: var(--text-primary); }

  .poster {
    width: calc(420 * var(--u));
    height: calc(594 * var(--u));
    background: var(--bg-page);
    margin: 20px auto;
    padding: calc(8 * var(--u)) calc(12 * var(--u));
    display: grid;
    grid-template-rows: auto 1fr auto auto;
    /* 10 mm row-gap → 38 px in print mode (--u: 1mm), comfortably inside
       the [30, 50] px target for measure's gap-to-footer-strip gate. */
    gap: calc(10 * var(--u));
    box-shadow: 0 0 40px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
  }
  .poster::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0;
    height: calc(4 * var(--u));
    background: linear-gradient(90deg, var(--accent-deep), var(--accent), var(--gold));
  }

  /* ---- header ---- */
  .header {
    display: grid;
    grid-template-columns: 1fr;
    align-items: center;
    gap: calc(8 * var(--u));
    padding: calc(2 * var(--u)) calc(2 * var(--u)) calc(4 * var(--u));
    border-bottom: calc(1 * var(--u)) solid var(--accent);
  }
  .title-block { min-width: 0; }
  .title {
    font-family: var(--font-sans);
    font-weight: 800;
    font-size: calc(20 * var(--u));
    line-height: 1.08;
    color: var(--accent-deep);
    letter-spacing: -0.4px;
  }
  .title .accent { color: var(--gold); }
  .subtitle {
    font-family: var(--font-sans);
    font-size: calc(9 * var(--u));
    color: var(--text-muted);
    margin-top: calc(1 * var(--u));
  }
  .authors {
    font-family: var(--font-sans);
    font-size: calc(8 * var(--u));
    color: var(--accent);
    font-weight: 600;
    margin-top: calc(2 * var(--u));
  }
  .qr-block { display: flex; flex-direction: column; align-items: center; gap: calc(1 * var(--u)); }
  .qr-block img {
    width: calc(28 * var(--u));
    height: calc(28 * var(--u));
    border: calc(0.5 * var(--u)) solid var(--accent);
    border-radius: calc(1.5 * var(--u));
    background: white;
    padding: calc(0.5 * var(--u));
  }
  .qr-label { font-family: var(--font-sans); font-size: calc(6 * var(--u)); color: var(--accent); font-weight: 600; }

  /* ---- body (single column, single card filling the row) ---- */
  .body-grid {
    display: grid;
    grid-template-columns: 1fr;
    min-height: 0;
  }
  .column {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .card {
    background: var(--bg-card);
    border-radius: calc(2 * var(--u));
    padding: calc(6 * var(--u)) calc(8 * var(--u));
    border: calc(0.5 * var(--u)) solid var(--border-soft);
    box-shadow: 0 calc(1 * var(--u)) calc(2 * var(--u)) rgba(0,0,0,0.04);
    border-left: calc(3 * var(--u)) solid var(--accent);
    flex: 1;
    display: flex;
    flex-direction: column;
  }
  .section-title {
    font-family: var(--font-sans);
    font-weight: 700;
    font-size: calc(11 * var(--u));
    color: var(--accent-deep);
    margin-bottom: calc(2 * var(--u));
  }
  .body-text {
    font-family: var(--font-serif);
    font-size: calc(7 * var(--u));
    line-height: 1.35;
    color: var(--text-primary);
  }
  .body-text + .body-text { margin-top: calc(1.5 * var(--u)); }
  .keyword { color: var(--accent); font-weight: 700; }
  .card ul { padding-left: calc(11 * var(--u)); margin-top: calc(1 * var(--u)); }
  .card li { font-family: var(--font-serif); font-size: calc(7 * var(--u)); line-height: 1.35; margin-top: calc(0.5 * var(--u)); }

  .figure { margin-top: calc(3 * var(--u)); display: flex; flex-direction: column; align-items: center; }
  /* 80% width on an AR=2 figure → tall figures would be too big, but
     AR=2 wide fits comfortably AND satisfies polish Gate A's wide-min
     ratio (must be >= 65% of card width). */
  /* Natural aspect ratio (AR=2), no max-height cap: the figure carries
     real vertical space instead of leaving the card half-blank, and
     stays >= 65% width so polish Gate A doesn't flag it as too small. */
  .figure img { width: 85%; display: block; }
  .figure .caption { font-family: var(--font-sans); font-size: calc(6 * var(--u)); color: var(--text-muted); margin-top: calc(1 * var(--u)); text-align: center; line-height: 1.2; }

  /* ---- footer-strip ---- */
  .footer-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: calc(4 * var(--u));
    background: var(--accent-deep);
    color: white;
    border-radius: calc(2 * var(--u));
    padding: calc(3 * var(--u)) calc(5 * var(--u));
  }
  .hs-stat { display: flex; flex-direction: column; align-items: center; gap: calc(0.5 * var(--u)); }
  .hs-stat .num {
    font-family: var(--font-sans);
    font-weight: 800;
    font-size: calc(12 * var(--u));
    color: var(--gold);
    white-space: nowrap;
  }
  .hs-stat .lbl {
    font-family: var(--font-sans);
    font-size: calc(6 * var(--u));
    color: var(--accent-light);
    text-align: center;
  }

  /* ---- footer ---- */
  .footer {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: calc(8 * var(--u));
    font-family: var(--font-sans);
    font-size: calc(6 * var(--u));
    color: var(--text-muted);
    padding: calc(1 * var(--u)) calc(2 * var(--u));
  }
  .footer .right { color: var(--accent); font-weight: 600; }

  /* =========================================================
     PRINT OVERRIDE — KEEP LAST so it wins source-order ties.
     Without `:root { --u: 1mm }` the poster keeps the 1.6 px /
     unit screen scale and renders into ~42 % of A2 — the column
     spread gate would still pass on the shrunken poster, but
     the printed PDF would show a giant gray border. measure
     hard-fails this via `--min-canvas-fill`. The explicit
     `html, body { height: 100% }` block dodges Chromium's
     well-known "round content height to next page" quirk that
     otherwise emits a phantom second page.
     ========================================================= */
  @media print {
    :root { --u: 1mm; }
    html, body {
      background: white;
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }
    .poster { margin: 0; box-shadow: none; page-break-after: avoid; }
  }
</style>
</head>
<body>

<div class="poster" data-measure-role="poster">

  <header class="header" data-measure-role="header">
    <div class="title-block">
      <h1 class="title">ARM–EBM equivalence <span class="accent">reproduces exactly</span></h1>
      <div class="subtitle">Independent CPU reproduction of Blondel et al. (ICML 2026, arXiv:2512.15605)</div>
      <div class="authors">Exact tabular checks &middot; V=8 &middot; T=4 &middot; 585 EOS responses + 4,096 fixed-length sequences</div>
    </div>
  </header>

  <div class="body-grid" data-measure-role="body">
    <div class="column" data-measure-role="column">

      <div class="card" data-measure-role="card">
        <div class="section-title">Outcome</div>
        <p class="body-text">
          The paper's core finite-state result holds in an exhaustive CPU implementation. Mapping arbitrary variable-length EBM rewards backwards into EOS-aware ARM logits reproduces every response probability to <span class="keyword">1.78e-15</span>; the ARM&rarr;reward&rarr;ARM fixed-point residual is <span class="keyword">6.66e-16</span>.
        </p>

        <div class="figure">
          <img alt="ARM EBM bijection and soft Bellman mapping" src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 480 180'><rect width='480' height='180' rx='14' fill='%23F6F3F0'/><rect x='24' y='48' width='120' height='74' rx='10' fill='%232D5F8B'/><text x='84' y='78' text-anchor='middle' font-family='sans-serif' font-size='16' fill='white' font-weight='bold'>EBM</text><text x='84' y='100' text-anchor='middle' font-family='sans-serif' font-size='10' fill='white'>global reward R(y)</text><rect x='336' y='48' width='120' height='74' rx='10' fill='%232D5F8B'/><text x='396' y='78' text-anchor='middle' font-family='sans-serif' font-size='16' fill='white' font-weight='bold'>ARM</text><text x='396' y='100' text-anchor='middle' font-family='sans-serif' font-size='10' fill='white'>local logits q(s,a)</text><path d='M150 72 H326' stroke='%23C9A24A' stroke-width='5'/><path d='M326 72 l-14 -9 v18 z' fill='%23C9A24A'/><path d='M330 104 H154' stroke='%232D5F8B' stroke-width='3'/><path d='M154 104 l14 -8 v16 z' fill='%232D5F8B'/><text x='240' y='55' text-anchor='middle' font-family='sans-serif' font-size='11' fill='%231F4566' font-weight='bold'>q = r + future soft value</text><text x='240' y='128' text-anchor='middle' font-family='sans-serif' font-size='11' fill='%231F4566'>r = q - next-state value</text><text x='240' y='160' text-anchor='middle' font-family='sans-serif' font-size='12' fill='%23C9A24A' font-weight='bold'>max log-probability error 1.78e-15</text></svg>">
          <div class="caption">The exact backward map is the acyclic soft Bellman equation; its inverse removes the next-state log-partition.</div>
        </div>

        <div class="section-title" style="margin-top: calc(4 * var(--u))">What was tested</div>
        <ul>
          <li><b>Claim 1.</b> Exact EBM&rarr;ARM probability preservation on all 585 valid variable-length EOS responses.</li>
          <li><b>Claim 2.</b> Soft Bellman ARM&rarr;reward&rarr;ARM round trip over every valid state and action.</li>
          <li><b>Claim 3.</b> Analytical ARM and EBM NLL gaps are exactly zero on both paper sequence spaces.</li>
          <li><b>Claim 4.</b> 80/80 directional KL inequalities pass; worst observed KL/bound ratio: 0.0716.</li>
          <li><b>Cost mechanism.</b> Sequence enumeration grows from 4,096 paths at T=4 to 262,144 at T=6 for V=8.</li>
        </ul>

        <div class="section-title" style="margin-top: calc(4 * var(--u))">Interpretation</div>
        <p class="body-text">
          The ARM's next-token logits encode a <em>future soft value</em>. In the fixed-horizon tree, the backwards Bellman recursion supplies that value exactly, so ancestral ARM sampling and globally normalized EBM sampling define the same distribution.
        </p>
        <ul>
          <li><b>Scope.</b> This is an exact function-space reproduction of the paper's Appendix-C synthetic special case.</li>
          <li><b>Not claimed.</b> It does not reproduce causal/non-causal Transformer training curves or establish LLM-scale behavior.</li>
          <li><b>Cost.</b> The complete reported CPU run, including both NLL optimizations, took 9.1 seconds.</li>
        </ul>

        <div class="section-title" style="margin-top: calc(4 * var(--u))">Reproduction bundle</div>
        <ul>
          <li><b>Code.</b> Exact ARM/EBM mapping, optimization runner, and regression tests.</li>
          <li><b>Evidence.</b> Raw NLL and KL-bound CSVs, JSON report, and an HTML convergence plot.</li>
          <li><b>Rerun.</b> <code>./reproduce_cpu.sh</code> recreates the environment and all reported outputs.</li>
        </ul>

        <div class="section-title" style="margin-top: calc(2 * var(--u))">Machine-readable evidence</div>
        <ul>
          <li><b>report.json</b> &mdash; configuration, environment, verdicts, and headline errors for all five claims.</li>
          <li><b>training_history.csv</b> &mdash; exact ARM and EBM expected-NLL gaps through 3,000 updates.</li>
          <li><b>kl_bound_trials.csv</b> &mdash; both directional KL values, infinity-norm error, bound, and ratio for every trial.</li>
        </ul>
      </div>

    </div>
  </div>

  <div class="footer-strip" data-measure-role="footer-strip">
    <div class="hs-stat">
      <div class="num">1.78e-15</div>
      <div class="lbl">maximum EBM/ARM log-probability error</div>
    </div>
    <div class="hs-stat">
      <div class="num">80 / 80</div>
      <div class="lbl">directional KL-bound checks passed</div>
    </div>
    <div class="hs-stat">
      <div class="num">~10 s</div>
      <div class="lbl">complete CPU run time</div>
    </div>
  </div>

  <footer class="footer" data-measure-role="footer">
    <div>Exact finite-state reproduction &middot; ICML 2026 challenge logbook</div>
    <div class="right">arXiv:2512.15605</div>
  </footer>

</div>

</body>
</html>

````
