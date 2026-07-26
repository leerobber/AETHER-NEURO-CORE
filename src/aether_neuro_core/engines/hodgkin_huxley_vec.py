"""Vectorized Hodgkin-Huxley (1952) engine.

Classic squid-axon HH kinetics using the widely-used NEURON `hh.mod`
parameterization (absolute membrane voltage in mV, resting ~ -65 mV),
vectorized across a batch dimension so many independent neuron instances
(e.g. one per ion-channel variant profile in a parameter sweep) integrate
in lockstep as NumPy array operations.

Two integrators are exposed:
  - `step` / `run(method="euler_maruyama")`: forward Euler on the deterministic
    system plus an additive Euler-Maruyama voltage-noise term. Use this for
    stochastic spike-train / inter-spike-interval variability studies.
  - `step_rk4` / `run(method="rk4")`: classical RK4 on the deterministic
    system only (no noise). Higher accuracy per step for pure spike-shape or
    threshold studies where stochasticity isn't wanted.

Reference rate constants (alpha/beta for m, h, n) and the classic conductance
set (g_Na=120, g_K=36, g_leak=0.3 mS/cm^2; E_Na=50, E_K=-77, E_leak=-54.387 mV)
match the standard squid-axon parameterization used across HH textbook
implementations (e.g. Gerstner & Kistler, *Neuronal Dynamics*).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def _vtrap(x: np.ndarray, y: float) -> np.ndarray:
    """Numerically stable x / (exp(x/y) - 1), with the removable singularity at x=0
    handled via its analytic limit y*(1 - x/(2y)) (standard "vtrap" trick from the
    original Hines HH implementation).
    """
    ratio = x / y
    small = np.abs(ratio) < 1e-6
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denom = np.exp(ratio) - 1.0
        big_branch = x / denom
    small_branch = y * (1.0 - ratio / 2.0)
    return np.where(small, small_branch, big_branch)


def detect_spikes(v_trace: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Detect upward threshold crossings (spikes) in a (time, batch) or (time,) voltage trace.

    Returns a boolean array of the same shape as `v_trace`, True at the first
    sample of each crossing from below-threshold to at-or-above-threshold.
    """
    v_trace = np.asarray(v_trace)
    above = v_trace >= threshold
    crossed = np.zeros_like(above, dtype=bool)
    crossed[1:] = above[1:] & ~above[:-1]
    return crossed


class HighThroughputHHEngine:
    """Vectorized 4-state (V, m, h, n) Hodgkin-Huxley engine.

    Conductances (`g_na`, `g_k`, `g_leak`) and kinetic shifts (`na_v_shift_mv`,
    `k_v_shift_mv`) may each be passed as a scalar (applied to every neuron in
    the batch) or as a per-neuron array of shape `(batch_size,)`, which is how
    a parameter sweep across variant profiles is expressed.
    """

    def __init__(
        self,
        batch_size: int,
        c_m: ArrayLike = 1.0,
        g_na: ArrayLike = 120.0,
        g_k: ArrayLike = 36.0,
        g_leak: ArrayLike = 0.3,
        e_na: ArrayLike = 50.0,
        e_k: ArrayLike = -77.0,
        e_leak: ArrayLike = -54.387,
        na_v_shift_mv: ArrayLike = 0.0,
        k_v_shift_mv: ArrayLike = 0.0,
        dt: float = 0.01,
        dtype: np.dtype = np.float32,
        seed: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.batch_size = batch_size
        self.dtype = dtype
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        self.c_m = self._broadcast(c_m)
        self.g_na = self._broadcast(g_na)
        self.g_k = self._broadcast(g_k)
        self.g_leak = self._broadcast(g_leak)
        self.e_na = self._broadcast(e_na)
        self.e_k = self._broadcast(e_k)
        self.e_leak = self._broadcast(e_leak)
        self.na_v_shift_mv = self._broadcast(na_v_shift_mv)
        self.k_v_shift_mv = self._broadcast(k_v_shift_mv)

        self.V: np.ndarray
        self.m: np.ndarray
        self.h: np.ndarray
        self.n: np.ndarray
        self.reset()

    def _broadcast(self, value: ArrayLike) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(self.batch_size, float(arr), dtype=np.float64)
        if arr.shape != (self.batch_size,):
            raise ValueError(
                f"per-neuron parameter must have shape ({self.batch_size},), got {arr.shape}"
            )
        return arr

    def _rates(self, V: np.ndarray):
        v_na = V - self.na_v_shift_mv
        v_k = V - self.k_v_shift_mv

        alpha_m = 0.1 * _vtrap(-(v_na + 40.0), 10.0)
        beta_m = 4.0 * np.exp(-(v_na + 65.0) / 18.0)
        alpha_h = 0.07 * np.exp(-(v_na + 65.0) / 20.0)
        beta_h = 1.0 / (np.exp(-(v_na + 35.0) / 10.0) + 1.0)
        alpha_n = 0.01 * _vtrap(-(v_k + 55.0), 10.0)
        beta_n = 0.125 * np.exp(-(v_k + 65.0) / 80.0)
        return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

    def rates(self, V: np.ndarray):
        """Public accessor for the (alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n) rate functions."""
        return self._rates(V)

    def steady_state_gates(self, V: np.ndarray):
        """Return (m_inf, h_inf, n_inf) for the given voltage array."""
        am, bm, ah, bh, an, bn = self._rates(V)
        return am / (am + bm), ah / (ah + bh), an / (an + bn)

    def reset(self, v_init: ArrayLike = -65.0) -> None:
        """Reset all neurons in the batch to their steady-state gating values at `v_init`."""
        V = self._broadcast(v_init)
        m_inf, h_inf, n_inf = self.steady_state_gates(V)
        self.V = V.astype(self.dtype)
        self.m = m_inf.astype(self.dtype)
        self.h = h_inf.astype(self.dtype)
        self.n = n_inf.astype(self.dtype)

    def membrane_current(self, V: np.ndarray, m: np.ndarray, h: np.ndarray, n: np.ndarray):
        i_na = self.g_na * m**3 * h * (V - self.e_na)
        i_k = self.g_k * n**4 * (V - self.e_k)
        i_leak = self.g_leak * (V - self.e_leak)
        return i_na, i_k, i_leak

    def _derivatives(self, V, m, h, n, i_ext: ArrayLike):
        am, bm, ah, bh, an, bn = self._rates(V)
        i_na, i_k, i_leak = self.membrane_current(V, m, h, n)
        dV = (np.asarray(i_ext, dtype=np.float64) - i_na - i_k - i_leak) / self.c_m
        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n
        return dV, dm, dh, dn

    def step(self, i_ext: ArrayLike, dt: Optional[float] = None, sigma: float = 0.0) -> np.ndarray:
        """Advance one forward-Euler step, with an optional Euler-Maruyama voltage-noise term.

        `sigma` is the noise intensity (same units as membrane current density);
        the voltage kick added is `sigma * sqrt(dt) * N(0, 1) / c_m`.
        """
        step_dt = self.dt if dt is None else dt
        dV, dm, dh, dn = self._derivatives(self.V, self.m, self.h, self.n, i_ext)

        noise = 0.0
        if sigma:
            noise = sigma * np.sqrt(step_dt) * self.rng.standard_normal(self.batch_size) / self.c_m

        self.V = (self.V + dV * step_dt + noise).astype(self.dtype)
        self.m = np.clip(self.m + dm * step_dt, 0.0, 1.0).astype(self.dtype)
        self.h = np.clip(self.h + dh * step_dt, 0.0, 1.0).astype(self.dtype)
        self.n = np.clip(self.n + dn * step_dt, 0.0, 1.0).astype(self.dtype)
        return self.V

    def step_rk4(self, i_ext: ArrayLike, dt: Optional[float] = None) -> np.ndarray:
        """Advance one classical RK4 step on the deterministic system (no noise)."""
        step_dt = self.dt if dt is None else dt

        def deriv(V, m, h, n):
            return self._derivatives(V, m, h, n, i_ext)

        V0, m0, h0, n0 = (
            self.V.astype(np.float64),
            self.m.astype(np.float64),
            self.h.astype(np.float64),
            self.n.astype(np.float64),
        )
        k1 = deriv(V0, m0, h0, n0)
        k2 = deriv(
            V0 + 0.5 * step_dt * k1[0],
            m0 + 0.5 * step_dt * k1[1],
            h0 + 0.5 * step_dt * k1[2],
            n0 + 0.5 * step_dt * k1[3],
        )
        k3 = deriv(
            V0 + 0.5 * step_dt * k2[0],
            m0 + 0.5 * step_dt * k2[1],
            h0 + 0.5 * step_dt * k2[2],
            n0 + 0.5 * step_dt * k2[3],
        )
        k4 = deriv(
            V0 + step_dt * k3[0],
            m0 + step_dt * k3[1],
            h0 + step_dt * k3[2],
            n0 + step_dt * k3[3],
        )

        self.V = (V0 + (step_dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])).astype(self.dtype)
        self.m = np.clip(
            m0 + (step_dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 0.0, 1.0
        ).astype(self.dtype)
        self.h = np.clip(
            h0 + (step_dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]), 0.0, 1.0
        ).astype(self.dtype)
        self.n = np.clip(
            n0 + (step_dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]), 0.0, 1.0
        ).astype(self.dtype)
        return self.V

    def run(
        self,
        n_steps: int,
        i_ext: ArrayLike,
        dt: Optional[float] = None,
        sigma: float = 0.0,
        method: str = "euler_maruyama",
    ) -> np.ndarray:
        """Integrate for `n_steps`, returning a (n_steps, batch_size) voltage trace.

        `i_ext` may be a scalar/per-neuron array (held constant across time) or
        a (n_steps, batch_size)-shaped array of a time-varying stimulus.
        """
        if method not in ("euler_maruyama", "rk4"):
            raise ValueError("method must be 'euler_maruyama' or 'rk4'")
        if method == "rk4" and sigma:
            raise ValueError("RK4 integrator does not support stochastic noise (sigma must be 0)")

        i_ext_arr = np.asarray(i_ext, dtype=np.float64)
        time_varying = i_ext_arr.ndim >= 1 and i_ext_arr.shape[0] == n_steps

        trace = np.empty((n_steps, self.batch_size), dtype=np.float64)
        for t in range(n_steps):
            i_t = i_ext_arr[t] if time_varying else i_ext_arr
            if method == "euler_maruyama":
                trace[t] = self.step(i_t, dt=dt, sigma=sigma)
            else:
                trace[t] = self.step_rk4(i_t, dt=dt)
        return trace
