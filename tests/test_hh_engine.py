import numpy as np
import pytest

from aether_neuro_core.engines.hodgkin_huxley_vec import HighThroughputHHEngine, detect_spikes


def test_resting_potential_is_stable_with_no_input():
    """With zero external current, a neuron initialized at rest should stay near rest."""
    engine = HighThroughputHHEngine(batch_size=4, dt=0.01, seed=0)
    trace = engine.run(n_steps=2000, i_ext=0.0, method="rk4")
    assert np.all(np.abs(trace[-1] - (-65.0)) < 1.0)
    assert np.all(np.isfinite(trace))


def test_suprathreshold_current_produces_spikes():
    """A standard current step well above rheobase should produce multiple spikes."""
    engine = HighThroughputHHEngine(batch_size=2, dt=0.01, seed=0)
    trace = engine.run(n_steps=5000, i_ext=10.0, method="rk4")  # 50 ms at dt=0.01ms
    spikes = detect_spikes(trace, threshold=0.0)
    spike_counts = spikes.sum(axis=0)
    assert np.all(spike_counts >= 2), f"expected repetitive spiking, got counts={spike_counts}"


def test_subthreshold_current_does_not_spike():
    """A small current step should depolarize slightly but not cross threshold."""
    engine = HighThroughputHHEngine(batch_size=2, dt=0.01, seed=0)
    trace = engine.run(n_steps=2000, i_ext=0.5, method="rk4")
    spikes = detect_spikes(trace, threshold=0.0)
    assert not np.any(spikes)


def test_batch_independence():
    """Different neurons in the same batch with different currents must evolve independently."""
    engine = HighThroughputHHEngine(batch_size=2, dt=0.01, seed=0)
    trace = engine.run(n_steps=3000, i_ext=np.array([0.0, 15.0]), method="rk4")
    spikes = detect_spikes(trace, threshold=0.0)
    assert spikes[:, 0].sum() == 0
    assert spikes[:, 1].sum() >= 1


def test_conductance_scaling_changes_excitability():
    """Halving g_na should raise the effective threshold, reducing spike count for the same drive."""
    baseline = HighThroughputHHEngine(batch_size=1, dt=0.01, seed=0)
    reduced = HighThroughputHHEngine(batch_size=1, g_na=60.0, dt=0.01, seed=0)

    i_ext = 8.0
    n_steps = 4000
    baseline_spikes = detect_spikes(baseline.run(n_steps, i_ext, method="rk4"), threshold=0.0).sum()
    reduced_spikes = detect_spikes(reduced.run(n_steps, i_ext, method="rk4"), threshold=0.0).sum()

    assert reduced_spikes <= baseline_spikes


def test_euler_maruyama_noise_produces_isi_variability():
    """With identical seeds-off noise, added stochastic drive should change spike timing."""
    engine_a = HighThroughputHHEngine(batch_size=1, dt=0.01, seed=1)
    engine_b = HighThroughputHHEngine(batch_size=1, dt=0.01, seed=2)

    trace_a = engine_a.run(n_steps=3000, i_ext=6.0, sigma=15.0, method="euler_maruyama")
    trace_b = engine_b.run(n_steps=3000, i_ext=6.0, sigma=15.0, method="euler_maruyama")

    assert np.isfinite(trace_a).all()
    assert np.isfinite(trace_b).all()
    assert not np.allclose(trace_a, trace_b)


def test_rk4_rejects_nonzero_sigma():
    engine = HighThroughputHHEngine(batch_size=1, dt=0.01, seed=0)
    with pytest.raises(ValueError):
        engine.run(n_steps=10, i_ext=0.0, sigma=1.0, method="rk4")


def test_reset_returns_to_steady_state():
    engine = HighThroughputHHEngine(batch_size=3, dt=0.01, seed=0)
    engine.run(n_steps=1000, i_ext=10.0, method="rk4")
    engine.reset(v_init=-65.0)
    assert np.allclose(engine.V, -65.0)
    m_inf, h_inf, n_inf = engine.steady_state_gates(np.full(3, -65.0))
    assert np.allclose(engine.m, m_inf, atol=1e-6)
    assert np.allclose(engine.h, h_inf, atol=1e-6)
    assert np.allclose(engine.n, n_inf, atol=1e-6)


def test_per_neuron_parameter_shape_validation():
    with pytest.raises(ValueError):
        HighThroughputHHEngine(batch_size=3, g_na=np.array([1.0, 2.0]))


def test_detect_spikes_shapes_and_edges():
    trace = np.array([-70.0, -70.0, 5.0, 5.0, -70.0, 5.0])
    spikes = detect_spikes(trace, threshold=0.0)
    assert spikes.tolist() == [False, False, True, False, False, True]
