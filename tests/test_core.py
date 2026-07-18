import numpy as np

from repro_arm_ebm.core import (
    all_sequences,
    all_variable_sequences,
    arm_log_probs,
    ebm_log_probs,
    inverse_map_arm_to_rewards,
    inverse_variable_arm_to_rewards,
    map_ebm_to_arm,
    map_rewards_to_arm,
    map_variable_rewards_to_arm,
    max_finite_table_distance,
    max_table_distance,
    variable_arm_log_probs,
    variable_sequence_rewards,
    zero_variable_tables,
    zero_logits,
)


def test_ebm_to_arm_preserves_every_sequence_probability() -> None:
    vocab_size, horizon = 3, 3
    sequences = all_sequences(vocab_size, horizon)
    scores = np.linspace(-1.5, 2.5, len(sequences))
    q = map_ebm_to_arm(scores, vocab_size, horizon)
    assert np.max(np.abs(ebm_log_probs(scores) - arm_log_probs(q, sequences, vocab_size))) < 1e-12


def test_arm_reward_arm_round_trip_is_exact() -> None:
    rng = np.random.default_rng(0)
    q = [rng.normal(size=shape.shape) for shape in zero_logits(3, 4)]
    assert max_table_distance(q, map_rewards_to_arm(inverse_map_arm_to_rewards(q))) < 1e-12


def test_variable_length_eos_bijection_preserves_all_probabilities() -> None:
    vocab_size, max_horizon = 3, 4
    rng = np.random.default_rng(1)
    responses = all_variable_sequences(vocab_size, max_horizon)
    rewards = zero_variable_tables(vocab_size, max_horizon)
    for table in rewards:
        table[np.isfinite(table)] = rng.normal(size=np.isfinite(table).sum())
    q = map_variable_rewards_to_arm(rewards)
    ebm_logp = ebm_log_probs(variable_sequence_rewards(rewards, responses, vocab_size))
    arm_logp = variable_arm_log_probs(q, responses, vocab_size)
    assert len(responses) == sum(vocab_size**k for k in range(max_horizon))
    assert np.max(np.abs(ebm_logp - arm_logp)) < 1e-12


def test_variable_length_bellman_round_trip_is_exact() -> None:
    vocab_size, max_horizon = 3, 4
    rng = np.random.default_rng(2)
    q = zero_variable_tables(vocab_size, max_horizon)
    for table in q:
        table[np.isfinite(table)] = rng.normal(size=np.isfinite(table).sum())
    recovered = map_variable_rewards_to_arm(inverse_variable_arm_to_rewards(q))
    assert max_finite_table_distance(q, recovered) < 1e-12
