"""Run an independent CPU reproduction of the paper's finite-state claims."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from repro_arm_ebm.core import (
    all_sequences,
    all_variable_sequences,
    arm_log_probs,
    arm_loss_and_gradient,
    centre_actions,
    context_ids,
    ebm_log_probs,
    ebm_loss_and_gradient,
    entropy,
    inverse_map_arm_to_rewards,
    inverse_variable_arm_to_rewards,
    kl_from_log_probs,
    map_ebm_to_arm,
    map_rewards_to_arm,
    map_variable_rewards_to_arm,
    max_finite_table_distance,
    max_table_distance,
    prefix_target_counts,
    softmax,
    zipf_target,
    variable_arm_log_probs,
    variable_sequence_rewards,
    zero_variable_tables,
    zero_logits,
)


class Adam:
    def __init__(self, values: list[np.ndarray] | np.ndarray, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.beta1, self.beta2, self.eps, self.step = 0.9, 0.999, 1e-8, 0
        initial = values if isinstance(values, list) else [values]
        self.m = [np.zeros_like(value) for value in initial]
        self.v = [np.zeros_like(value) for value in initial]

    def update(self, values: list[np.ndarray] | np.ndarray, gradients: list[np.ndarray] | np.ndarray) -> None:
        arrays = values if isinstance(values, list) else [values]
        grads = gradients if isinstance(gradients, list) else [gradients]
        self.step += 1
        for value, grad, moment, second in zip(arrays, grads, self.m, self.v, strict=True):
            moment *= self.beta1
            moment += (1 - self.beta1) * grad
            second *= self.beta2
            second += (1 - self.beta2) * grad * grad
            corrected_m = moment / (1 - self.beta1**self.step)
            corrected_v = second / (1 - self.beta2**self.step)
            value -= self.learning_rate * corrected_m / (np.sqrt(corrected_v) + self.eps)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot_html(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Small self-contained loss plot; the CSV remains the source data."""
    width, height, pad = 760, 420, 55
    values = [float(row["ebm_gap"]) for row in rows] + [float(row["arm_gap"]) for row in rows]
    positive = np.log10(np.maximum(values, 1e-16))
    low, high = float(positive.min()), float(positive.max())
    high = max(high, low + 1e-6)
    max_step = max(int(row["step"]) for row in rows)

    def points(key: str) -> str:
        output = []
        for row in rows:
            x = pad + (width - 2 * pad) * int(row["step"]) / max_step
            y = height - pad - (height - 2 * pad) * (np.log10(max(float(row[key]), 1e-16)) - low) / (high - low)
            output.append(f"{x:.1f},{y:.1f}")
        return " ".join(output)

    html = f"""<!doctype html><html><body><svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"ARM and EBM optimality gaps\">
<rect width=\"100%\" height=\"100%\" fill=\"white\"/><line x1=\"{pad}\" y1=\"{height-pad}\" x2=\"{width-pad}\" y2=\"{height-pad}\" stroke=\"black\"/><line x1=\"{pad}\" y1=\"{pad}\" x2=\"{pad}\" y2=\"{height-pad}\" stroke=\"black\"/>
<polyline fill=\"none\" stroke=\"#1565c0\" stroke-width=\"2\" points=\"{points('ebm_gap')}\"/><polyline fill=\"none\" stroke=\"#d84315\" stroke-width=\"2\" points=\"{points('arm_gap')}\"/>
<text x=\"{pad}\" y=\"24\" font-size=\"16\">Exact expected-risk optimality gap (log scale)</text><text x=\"{width-230}\" y=\"45\" fill=\"#1565c0\">EBM</text><text x=\"{width-160}\" y=\"45\" fill=\"#d84315\">ARM</text><text x=\"{width/2-18}\" y=\"{height-12}\">step</text>
</svg></body></html>"""
    path.write_text(html, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    sequences = all_sequences(args.vocab_size, args.horizon)
    num_sequences = len(sequences)
    rho = zipf_target(num_sequences)
    target_entropy = entropy(rho)
    counts = prefix_target_counts(rho, sequences, args.vocab_size)

    # Claims 1 and 2: variable-length, EOS-aware bijection and Bellman equation.
    variable_responses = all_variable_sequences(args.vocab_size, args.horizon)
    variable_rewards = zero_variable_tables(args.vocab_size, args.horizon)
    for table in variable_rewards:
        finite = np.isfinite(table)
        table[finite] = rng.normal(size=finite.sum())
    variable_q = map_variable_rewards_to_arm(variable_rewards)
    variable_scores = variable_sequence_rewards(
        variable_rewards, variable_responses, args.vocab_size
    )
    variable_ebm_logp = ebm_log_probs(variable_scores)
    variable_arm_logp = variable_arm_log_probs(
        variable_q, variable_responses, args.vocab_size
    )
    random_variable_q = zero_variable_tables(args.vocab_size, args.horizon)
    for table in random_variable_q:
        finite = np.isfinite(table)
        table[finite] = rng.normal(size=finite.sum())
    variable_round_trip = map_variable_rewards_to_arm(
        inverse_variable_arm_to_rewards(random_variable_q)
    )
    variable_bellman_residual = max_finite_table_distance(
        random_variable_q, variable_round_trip
    )

    # Fixed-length Appendix-C check, retained as an independent cross-check.
    random_scores = rng.normal(size=num_sequences)
    mapped_q = map_ebm_to_arm(random_scores, args.vocab_size, args.horizon)
    ebm_logp = ebm_log_probs(random_scores)
    arm_logp = arm_log_probs(mapped_q, sequences, args.vocab_size)
    random_q = [rng.normal(size=table.shape) for table in mapped_q]
    round_trip_q = map_rewards_to_arm(inverse_map_arm_to_rewards(random_q))
    bellman_residual = max_table_distance(random_q, round_trip_q)

    # Claim 3: direct tabular NLL optimization reaches the same entropy minimum.
    ebm_scores = rng.normal(scale=0.05, size=num_sequences)
    arm_q = [rng.normal(scale=0.05, size=table.shape) for table in zero_logits(args.vocab_size, args.horizon)]
    ebm_adam, arm_adam = Adam(ebm_scores, args.learning_rate), Adam(arm_q, args.learning_rate)
    history: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        ebm_loss, ebm_grad = ebm_loss_and_gradient(ebm_scores, rho)
        arm_loss, arm_grad = arm_loss_and_gradient(arm_q, rho, sequences, args.vocab_size, counts)
        ebm_adam.update(ebm_scores, ebm_grad)
        arm_adam.update(arm_q, arm_grad)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append({"step": step, "ebm_gap": max(ebm_loss - target_entropy, 0.0), "arm_gap": max(arm_loss - target_entropy, 0.0)})
    train_seconds = time.perf_counter() - started
    final_ebm_loss, _ = ebm_loss_and_gradient(ebm_scores, rho)
    final_arm_loss, _ = arm_loss_and_gradient(arm_q, rho, sequences, args.vocab_size, counts)

    # The function-space optimum is available analytically and reaches H(rho).
    analytic_ebm_scores = np.log(rho)
    analytic_arm_q = [
        np.log(target_counts / target_counts.sum(axis=1, keepdims=True))
        for target_counts in counts
    ]
    analytic_ebm_loss, _ = ebm_loss_and_gradient(analytic_ebm_scores, rho)
    analytic_arm_loss, _ = arm_loss_and_gradient(
        analytic_arm_q, rho, sequences, args.vocab_size, counts
    )
    analytic_probability_error = float(
        np.max(
            np.abs(
                ebm_log_probs(analytic_ebm_scores)
                - arm_log_probs(analytic_arm_q, sequences, args.vocab_size)
            )
        )
    )

    # Reproduce the paper's second enumerated sequence space (V=4, T=8)
    # analytically; optimization is unnecessary because the optimum is explicit.
    second_sequences = all_sequences(4, 8)
    second_rho = zipf_target(len(second_sequences))
    second_counts = prefix_target_counts(second_rho, second_sequences, 4)
    second_q = [
        np.log(target_counts / target_counts.sum(axis=1, keepdims=True))
        for target_counts in second_counts
    ]
    second_probability_error = float(
        np.max(
            np.abs(
                ebm_log_probs(np.log(second_rho))
                - arm_log_probs(second_q, second_sequences, 4)
            )
        )
    )

    # Compare the learned ARM with the ARM logits obtained by mapping the learned EBM.
    optimal_from_ebm = map_ebm_to_arm(ebm_scores, args.vocab_size, args.horizon)
    centred_distance = max_table_distance(centre_actions(arm_q), centre_actions(optimal_from_ebm))

    # Claim 4: exhaustive numerical checks in both KL directions.
    kl_rows: list[dict[str, float | int | str]] = []
    exact_q = map_ebm_to_arm(random_scores, args.vocab_size, args.horizon)
    for scale in (0.01, 0.05, 0.2, 0.5, 1.0):
        for trial in range(args.kl_trials):
            candidate = [table + rng.normal(scale=scale, size=table.shape) for table in exact_q]
            error = max_table_distance(candidate, exact_q)
            candidate_logp = arm_log_probs(candidate, sequences, args.vocab_size)
            kl_forward = kl_from_log_probs(candidate_logp, ebm_logp)
            kl_reverse = kl_from_log_probs(ebm_logp, candidate_logp)
            bound = 2 * args.horizon * error
            kl_rows.append({
                "scale": scale,
                "trial": trial,
                "linf_error": error,
                "kl_arm_to_ebm": kl_forward,
                "kl_ebm_to_arm": kl_reverse,
                "bound": bound,
                "max_ratio": max(kl_forward, kl_reverse) / bound if bound else 0.0,
            })

    # Claim 5: report the exact state-space growth responsible for EBM-to-ARM cost.
    complexity = []
    for horizon in range(1, args.horizon + 3):
        leaves = args.vocab_size**horizon
        contexts = sum(args.vocab_size**t for t in range(horizon))
        complexity.append({"vocab_size": args.vocab_size, "horizon": horizon, "complete_sequences": leaves, "context_action_logits": contexts * args.vocab_size})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "training_history.csv", history)
    write_csv(output_dir / "kl_bound_trials.csv", kl_rows)
    write_csv(output_dir / "complexity.csv", complexity)
    write_plot_html(output_dir / "training_plot.html", history)

    report: dict[str, object] = {
        "paper": "Autoregressive Language Models are Secretly Energy-Based Models",
        "scope": "Exact CPU verification in function space: variable-length EOS responses for Claims 1-2 and both fixed-horizon sequence spaces from Appendix C for Claim 3; not a finite-Transformer/LLM-scale replication.",
        "configuration": {"vocab_size": args.vocab_size, "horizon": args.horizon, "sequences": num_sequences, "steps": args.steps, "seed": args.seed, "device": "CPU"},
        "claim_1_bijection": {
            "setting": "Variable-length responses with EOS and maximum length T",
            "valid_responses": len(variable_responses),
            "max_abs_log_probability_difference": float(
                np.max(np.abs(variable_ebm_logp - variable_arm_logp))
            ),
            "fixed_length_cross_check": float(np.max(np.abs(ebm_logp - arm_logp))),
            "passed": bool(
                np.max(np.abs(variable_ebm_logp - variable_arm_logp)) < 1e-10
            ),
        },
        "claim_2_soft_bellman": {
            "max_abs_fixed_point_residual": variable_bellman_residual,
            "fixed_length_cross_check": bellman_residual,
            "passed": bool(variable_bellman_residual < 1e-10),
        },
        "claim_3_same_minima": {
            "target_entropy": target_entropy,
            "analytic_ebm_gap": analytic_ebm_loss - target_entropy,
            "analytic_arm_gap": analytic_arm_loss - target_entropy,
            "analytic_probability_error_v8_t4": analytic_probability_error,
            "analytic_probability_error_v4_t8": second_probability_error,
            "trained_ebm_final_gap": final_ebm_loss - target_entropy,
            "trained_arm_final_gap": final_arm_loss - target_entropy,
            "centred_arm_vs_mapped_ebm_linf": centred_distance,
            "training_seconds": train_seconds,
        },
        "claim_4_kl_bound": {
            "trials": len(kl_rows),
            "directions_per_trial": 2,
            "max_kl_over_bound": max(float(row["max_ratio"]) for row in kl_rows),
            "all_bounds_hold": all(
                max(float(row["kl_arm_to_ebm"]), float(row["kl_ebm_to_arm"]))
                <= float(row["bound"]) + 1e-12
                for row in kl_rows
            ),
        },
        "claim_5_complexity": {"complexity_table": complexity, "interpretation": "Enumerating every complete sequence grows as V^T; the backwards EBM-to-ARM sweep must visit this exponentially growing tree."},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--kl-trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="outputs/cpu_exact")
    return parser.parse_args()


if __name__ == "__main__":
    report = run(parse_args())
    print(json.dumps(report, indent=2))
