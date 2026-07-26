# AETHER-NEURO-CORE

AetherNeuroCore is a biophysical simulation framework for vectorized 4-state
Hodgkin-Huxley (1952) neuron models, with a Hines-style implicit solver for
multi-compartment (soma + dendrite tree) dynamics, and a small agent layer
that maps molecular metadata (UniProt accessions) onto simulation parameters.

The CPU/NumPy engines are real, tested code. A CUDA translation of the core
HH step exists in `src/aether_neuro_core/cuda_kernels/hh_solver.cu` but is
**unverified** — there's no GPU in the environment this was built in, so it
has never been compiled or run. See `docs/ARCHITECTURE.md` for the full
picture of what's implemented vs. aspirational.

## Install

```bash
pip install -e ".[test]"
```

## Quick start

```python
from aether_neuro_core.engines.hodgkin_huxley_vec import HighThroughputHHEngine

engine = HighThroughputHHEngine(batch_size=4, dt=0.01, seed=0)
trace = engine.run(n_steps=5000, i_ext=10.0, method="rk4")  # (n_steps, 4) voltage trace
```

```python
from aether_neuro_core.engines.multi_compartment import Compartment, MultiCompartmentNeuronEngine

soma = Compartment(name="soma", parent=None, diam_um=20.0, length_um=20.0)
dend = Compartment(name="dend", parent="soma", diam_um=2.0, length_um=200.0)
neuron = MultiCompartmentNeuronEngine([soma, dend], dt=0.01)

trace = neuron.run(n_steps=2000, i_ext=[12.0, 0.0])  # current injected into soma only
```

## Test

```bash
pytest -v
```
