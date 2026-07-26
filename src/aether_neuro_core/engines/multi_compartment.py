"""Multi-compartment neuron engine with implicit axial coupling.

Models a single neuron as a tree of cylindrical compartments (e.g. soma +
dendritic branches), each carrying its own Hodgkin-Huxley membrane state via
a `HighThroughputHHEngine` whose batch dimension is the compartment index.
Axial current between a compartment and its parent is:

    I_axial = g_axial * (V_parent - V_compartment)

Because the tree couples every compartment's voltage ODE to its neighbors,
advancing all compartments with a plain explicit step is only conditionally
stable for realistic axial conductances. Instead this engine treats the axial
(linear) coupling implicitly and the ionic (nonlinear) membrane currents
explicitly — a standard operator-split scheme for cable equations — and
solves the resulting per-step linear system with the tree-structured Gaussian
elimination described by Hines (1984): each compartment's equation only
involves itself, its parent, and its children, so eliminating leaves toward
the root and then back-substituting root-to-leaves solves the whole tree in
O(N) time, without ever assembling or inverting a dense/sparse matrix.

Axial conductance is derived from compartment geometry under the standard
simplifying cable-theory assumptions: uniform diameter, length taken as the
distance between compartment centers, uniform axial resistivity, end-effects
ignored. This is a textbook approximation (see e.g. Koch, *Biophysics of
Computation*), not a substitute for a validated morphology-reconstruction
pipeline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from aether_neuro_core.engines.hodgkin_huxley_vec import HighThroughputHHEngine


@dataclass
class Compartment:
    """A single cylindrical compartment in a multi-compartment neuron tree."""

    name: str
    parent: Optional[str]
    diam_um: float
    length_um: float
    ra_ohm_cm: float = 100.0
    c_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_leak: float = 0.3

    @property
    def area_cm2(self) -> float:
        """Lateral surface area of the cylinder, in cm^2 (end caps ignored)."""
        d_cm = self.diam_um * 1e-4
        l_cm = self.length_um * 1e-4
        return np.pi * d_cm * l_cm

    def axial_conductance_density(self) -> float:
        """Specific (per-membrane-area) axial conductance to this compartment's
        parent, in mS/cm^2, derived from uniform-cylinder cable theory:
        g/area = d / (4 * Ra * L^2).
        """
        d_cm = self.diam_um * 1e-4
        l_cm = self.length_um * 1e-4
        return d_cm / (4.0 * self.ra_ohm_cm * l_cm**2) * 1000.0  # S/cm^2 -> mS/cm^2


def _topological_order(parent_idx: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return (order, root_index) where `order` lists all node indices sorted by
    decreasing depth (leaves first, root last) — a valid elimination order for
    Hines' tree-Gaussian-elimination: every child is processed before its parent.
    """
    n = len(parent_idx)
    roots = np.where(parent_idx < 0)[0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root compartment, found {len(roots)}")
    root = int(roots[0])

    children: List[List[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parent_idx):
        if p >= 0:
            children[int(p)].append(i)

    depth = np.zeros(n, dtype=int)
    bfs_order = [root]
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for c in children[node]:
            depth[c] = depth[node] + 1
            bfs_order.append(c)
            queue.append(c)

    if len(bfs_order) != n:
        raise ValueError("Compartment tree is disconnected or contains a cycle")

    leaves_first = sorted(bfs_order, key=lambda i: depth[i], reverse=True)
    return np.array(leaves_first, dtype=int), root


class MultiCompartmentNeuronEngine:
    """A single neuron composed of HH compartments coupled by implicit axial current."""

    def __init__(self, compartments: List[Compartment], dt: float = 0.01, seed: Optional[int] = None):
        if not compartments:
            raise ValueError("At least one compartment is required")

        self.compartments = compartments
        self.names = [c.name for c in compartments]

        seen: set = set()
        duplicates: List[str] = []
        for name in self.names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                f"Duplicate compartment name(s): {', '.join(sorted(duplicates))}. "
                "Compartment names must be unique for name-based indexing and "
                "parent resolution to be unambiguous."
            )

        self._index = {name: i for i, name in enumerate(self.names)}
        n = len(compartments)

        parent_idx = np.full(n, -1, dtype=int)
        for i, comp in enumerate(compartments):
            if comp.parent is not None:
                if comp.parent not in self._index:
                    raise ValueError(f"Compartment '{comp.name}' has unknown parent '{comp.parent}'")
                parent_idx[i] = self._index[comp.parent]

        self.parent_idx = parent_idx
        self.topo_order, self.root = _topological_order(parent_idx)
        self.dt = dt

        # g_axial_density[i] couples compartment i to parent_idx[i]; unused/0 for the root.
        self.g_axial_density = np.array(
            [c.axial_conductance_density() if p >= 0 else 0.0 for c, p in zip(compartments, parent_idx)],
            dtype=np.float64,
        )

        self.hh = HighThroughputHHEngine(
            batch_size=n,
            c_m=np.array([c.c_m for c in compartments]),
            g_na=np.array([c.g_na for c in compartments]),
            g_k=np.array([c.g_k for c in compartments]),
            g_leak=np.array([c.g_leak for c in compartments]),
            dt=dt,
            dtype=np.float64,
            seed=seed,
        )

    def index_of(self, name: str) -> int:
        return self._index[name]

    def voltages(self) -> np.ndarray:
        return self.hh.V

    def reset(self, v_init: float = -65.0) -> None:
        self.hh.reset(v_init=v_init)

    def step(self, i_ext, dt: Optional[float] = None) -> np.ndarray:
        """Advance one implicit-axial / explicit-ionic step; returns the new voltage array.

        `i_ext` is a per-compartment external current density (uA/cm^2), scalar
        or array of shape `(n_compartments,)`.
        """
        step_dt = self.dt if dt is None else dt
        n = len(self.compartments)
        i_ext_arr = np.broadcast_to(np.asarray(i_ext, dtype=np.float64), (n,)).copy()

        hh = self.hh
        i_na, i_k, i_leak = hh.membrane_current(hh.V, hh.m, hh.h, hh.n)
        i_ion = i_na + i_k + i_leak

        c_over_dt = hh.c_m / step_dt
        diag = c_over_dt.copy()
        rhs = c_over_dt * hh.V.astype(np.float64) - i_ion + i_ext_arr

        for i, p in enumerate(self.parent_idx):
            if p >= 0:
                g = self.g_axial_density[i]
                diag[i] += g
                diag[p] += g

        alpha = np.zeros(n, dtype=np.float64)
        beta = np.zeros(n, dtype=np.float64)

        for i in self.topo_order:
            p = self.parent_idx[i]
            alpha[i] = rhs[i] / diag[i]
            if p >= 0:
                g = self.g_axial_density[i]
                beta[i] = g / diag[i]
                diag[p] -= g * beta[i]
                rhs[p] += g * alpha[i]

        v_new = np.empty(n, dtype=np.float64)
        for i in self.topo_order[::-1]:
            p = self.parent_idx[i]
            v_new[i] = alpha[i] if p < 0 else alpha[i] + beta[i] * v_new[p]

        # Explicit update of gating variables at the pre-step voltage (operator split:
        # axial coupling solved implicitly above, gating kinetics advanced explicitly).
        am, bm, ah, bh, an, bn = hh.rates(hh.V.astype(np.float64))
        hh.m = np.clip(hh.m + (am * (1.0 - hh.m) - bm * hh.m) * step_dt, 0.0, 1.0)
        hh.h = np.clip(hh.h + (ah * (1.0 - hh.h) - bh * hh.h) * step_dt, 0.0, 1.0)
        hh.n = np.clip(hh.n + (an * (1.0 - hh.n) - bn * hh.n) * step_dt, 0.0, 1.0)
        hh.V = v_new

        return hh.V

    def run(self, n_steps: int, i_ext, dt: Optional[float] = None) -> np.ndarray:
        """Integrate for `n_steps`, returning a (n_steps, n_compartments) voltage trace."""
        n = len(self.compartments)
        i_ext_arr = np.asarray(i_ext, dtype=np.float64)
        time_varying = i_ext_arr.ndim >= 1 and i_ext_arr.shape[0] == n_steps

        trace = np.empty((n_steps, n), dtype=np.float64)
        for t in range(n_steps):
            i_t = i_ext_arr[t] if time_varying else i_ext_arr
            trace[t] = self.step(i_t, dt=dt)
        return trace
