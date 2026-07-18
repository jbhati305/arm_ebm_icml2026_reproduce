"""Numerically stable tabular ARM/EBM operations for a fixed-length vocabulary.

The paper's Appendix C uses this fixed-length special case.  A sequence is a
length-T vector over {0, ..., V-1}; no EOS token is needed because every path
ends at the same horizon.  This lets us enumerate every outcome exactly.
"""

from __future__ import annotations

from itertools import product

import numpy as np


Array = np.ndarray


def logsumexp(x: Array, axis: int | None = None) -> Array:
    """Stable log(sum(exp(x)))."""
    maximum = np.max(x, axis=axis, keepdims=True)
    reduced = np.log(np.sum(np.exp(x - maximum), axis=axis, keepdims=True)) + maximum
    if axis is None:
        return np.asarray(reduced.squeeze())
    return np.squeeze(reduced, axis=axis)


def logsoftmax(x: Array, axis: int = -1) -> Array:
    return x - np.expand_dims(logsumexp(x, axis=axis), axis=axis)


def softmax(x: Array, axis: int = -1) -> Array:
    return np.exp(logsoftmax(x, axis=axis))


def all_sequences(vocab_size: int, horizon: int) -> Array:
    if vocab_size < 2 or horizon < 1:
        raise ValueError("vocab_size must be >= 2 and horizon must be >= 1")
    return np.asarray(list(product(range(vocab_size), repeat=horizon)), dtype=np.int64)


def all_variable_sequences(vocab_size: int, max_horizon: int) -> list[tuple[int, ...]]:
    """Enumerate the paper's valid responses, including EOS as the last token.

    EOS is represented by ``vocab_size``. A response has between one and
    ``max_horizon`` tokens including EOS, so it contains at most T-1 ordinary
    vocabulary tokens.
    """
    if vocab_size < 2 or max_horizon < 1:
        raise ValueError("vocab_size must be >= 2 and max_horizon must be >= 1")
    eos = vocab_size
    responses: list[tuple[int, ...]] = []
    for content_length in range(max_horizon):
        for prefix in product(range(vocab_size), repeat=content_length):
            responses.append((*prefix, eos))
    return responses


def prefix_ids(sequences: Array, vocab_size: int, time: int) -> Array:
    """Map each length-``time`` prefix to its base-V integer index."""
    if time == 0:
        return np.zeros(len(sequences), dtype=np.int64)
    powers = vocab_size ** np.arange(time - 1, -1, -1, dtype=np.int64)
    return sequences[:, :time] @ powers


def context_ids(sequences: Array, vocab_size: int) -> list[Array]:
    return [prefix_ids(sequences, vocab_size, t) for t in range(sequences.shape[1])]


def zero_logits(vocab_size: int, horizon: int) -> list[Array]:
    return [np.zeros((vocab_size**t, vocab_size), dtype=np.float64) for t in range(horizon)]


def zero_variable_tables(vocab_size: int, max_horizon: int) -> list[Array]:
    """Create state-action tables over V ordinary actions plus EOS.

    At depth T-1 only EOS is valid, matching equation (3) in the paper.
    """
    tables = [
        np.zeros((vocab_size**depth, vocab_size + 1), dtype=np.float64)
        for depth in range(max_horizon)
    ]
    tables[-1][:, :vocab_size] = -np.inf
    return tables


def copy_logits(q: list[Array]) -> list[Array]:
    return [values.copy() for values in q]


def arm_log_probs(q: list[Array], sequences: Array, vocab_size: int) -> Array:
    """Return log p_q(y) for every enumerated sequence y."""
    if len(q) != sequences.shape[1]:
        raise ValueError("one logit table is required per time step")
    result = np.zeros(len(sequences), dtype=np.float64)
    for time, table in enumerate(q):
        ids = prefix_ids(sequences, vocab_size, time)
        result += logsoftmax(table[ids], axis=1)[np.arange(len(sequences)), sequences[:, time]]
    return result


def ebm_log_probs(scores: Array) -> Array:
    return logsoftmax(np.asarray(scores, dtype=np.float64), axis=0)


def map_ebm_to_arm(scores: Array, vocab_size: int, horizon: int) -> list[Array]:
    """Implement q=M(r), putting the sequence score at the final transition.

    For t<T-1, r(s_t, a_t)=0 and q(s_t,a_t)=V_q(s_t+a_t).  At
    t=T-1, q equals the sequence reward.  The backwards sweep is the soft
    Bellman recursion in Proposition 1.
    """
    expected = vocab_size**horizon
    if np.asarray(scores).shape != (expected,):
        raise ValueError(f"expected {expected} sequence scores")
    q = zero_logits(vocab_size, horizon)
    q[-1] = np.asarray(scores, dtype=np.float64).reshape(vocab_size ** (horizon - 1), vocab_size).copy()
    for time in range(horizon - 2, -1, -1):
        future_values = logsumexp(q[time + 1], axis=1)
        q[time] = future_values.reshape(vocab_size**time, vocab_size)
    return q


def inverse_map_arm_to_rewards(q: list[Array]) -> list[Array]:
    """Implement r=M^{-1}(q) in the fixed-horizon special case."""
    rewards = zero_logits(q[0].shape[1], len(q))
    rewards[-1] = q[-1].copy()
    for time in range(len(q) - 1):
        next_values = logsumexp(q[time + 1], axis=1).reshape(q[time].shape)
        rewards[time] = q[time] - next_values
    return rewards


def map_rewards_to_arm(rewards: list[Array]) -> list[Array]:
    """General fixed-horizon version of q=M(r)."""
    q = zero_logits(rewards[0].shape[1], len(rewards))
    q[-1] = rewards[-1].copy()
    for time in range(len(rewards) - 2, -1, -1):
        q[time] = rewards[time] + logsumexp(q[time + 1], axis=1).reshape(rewards[time].shape)
    return q


def map_variable_rewards_to_arm(rewards: list[Array]) -> list[Array]:
    """Exact Proposition-1/3.2 mapping for variable lengths and EOS."""
    vocab_size = rewards[0].shape[1] - 1
    q = zero_variable_tables(vocab_size, len(rewards))
    q[-1] = rewards[-1].copy()
    q[-1][:, :vocab_size] = -np.inf
    for depth in range(len(rewards) - 2, -1, -1):
        q[depth][:, vocab_size] = rewards[depth][:, vocab_size]
        future = logsumexp(q[depth + 1], axis=1).reshape(vocab_size**depth, vocab_size)
        q[depth][:, :vocab_size] = rewards[depth][:, :vocab_size] + future
    return q


def inverse_variable_arm_to_rewards(q: list[Array]) -> list[Array]:
    """Exact inverse of :func:`map_variable_rewards_to_arm`."""
    vocab_size = q[0].shape[1] - 1
    rewards = zero_variable_tables(vocab_size, len(q))
    rewards[-1] = q[-1].copy()
    for depth in range(len(q) - 1):
        rewards[depth][:, vocab_size] = q[depth][:, vocab_size]
        future = logsumexp(q[depth + 1], axis=1).reshape(vocab_size**depth, vocab_size)
        rewards[depth][:, :vocab_size] = q[depth][:, :vocab_size] - future
    return rewards


def variable_sequence_rewards(
    rewards: list[Array], responses: list[tuple[int, ...]], vocab_size: int
) -> Array:
    result = np.zeros(len(responses), dtype=np.float64)
    for row, response in enumerate(responses):
        prefix_id = 0
        for depth, action in enumerate(response):
            result[row] += rewards[depth][prefix_id, action]
            if action != vocab_size:
                prefix_id = prefix_id * vocab_size + action
    return result


def variable_arm_log_probs(
    q: list[Array], responses: list[tuple[int, ...]], vocab_size: int
) -> Array:
    normalized = [logsoftmax(table, axis=1) for table in q]
    result = np.zeros(len(responses), dtype=np.float64)
    for row, response in enumerate(responses):
        prefix_id = 0
        for depth, action in enumerate(response):
            result[row] += normalized[depth][prefix_id, action]
            if action != vocab_size:
                prefix_id = prefix_id * vocab_size + action
    return result


def max_finite_table_distance(left: list[Array], right: list[Array]) -> float:
    """Maximum difference over valid entries, requiring matching infinities."""
    distances: list[float] = []
    for first, second in zip(left, right, strict=True):
        if not np.array_equal(np.isneginf(first), np.isneginf(second)):
            return float("inf")
        finite = np.isfinite(first) & np.isfinite(second)
        if np.any(finite):
            distances.append(float(np.max(np.abs(first[finite] - second[finite]))))
    return max(distances, default=0.0)


def centre_actions(q: list[Array]) -> list[Array]:
    return [table - table.mean(axis=1, keepdims=True) for table in q]


def max_table_distance(left: list[Array], right: list[Array]) -> float:
    return float(max(np.max(np.abs(a - b)) for a, b in zip(left, right, strict=True)))


def prefix_target_counts(rho: Array, sequences: Array, vocab_size: int) -> list[Array]:
    """rho(s,a) for every teacher-forced state-action pair."""
    counts = zero_logits(vocab_size, sequences.shape[1])
    for time, table in enumerate(counts):
        np.add.at(table, (prefix_ids(sequences, vocab_size, time), sequences[:, time]), rho)
    return counts


def arm_loss_and_gradient(q: list[Array], rho: Array, sequences: Array, vocab_size: int, counts: list[Array]) -> tuple[float, list[Array]]:
    logp = arm_log_probs(q, sequences, vocab_size)
    gradients: list[Array] = []
    for table, target_counts in zip(q, counts, strict=True):
        mass = target_counts.sum(axis=1, keepdims=True)
        gradients.append(mass * softmax(table, axis=1) - target_counts)
    return float(-np.dot(rho, logp)), gradients


def ebm_loss_and_gradient(scores: Array, rho: Array) -> tuple[float, Array]:
    logp = ebm_log_probs(scores)
    return float(-np.dot(rho, logp)), np.exp(logp) - rho


def kl_from_log_probs(logp: Array, logq: Array) -> float:
    p = np.exp(logp)
    return float(np.dot(p, logp - logq))


def zipf_target(num_sequences: int, exponent: float = 1.15) -> Array:
    ranks = np.arange(1, num_sequences + 1, dtype=np.float64)
    weights = ranks ** (-exponent)
    return weights / weights.sum()


def entropy(probabilities: Array) -> float:
    return float(-np.dot(probabilities, np.log(probabilities)))
