# AetherNeuroCore Architecture

## What this is

A vectorized, pure-Python/NumPy Hodgkin-Huxley (1952) simulation engine, a
Hines-style implicit multi-compartment tree solver built on top of it, and a
small async agent layer that maps molecular metadata (UniProt accessions) to
simulation parameters. Everything under `src/aether_neuro_core/engines/` and
`src/aether_neuro_core/agents/` is real, tested code you can run today.

## What this isn't (yet)

The original project brief described a "zero-overhead, GPU-accelerated
runtime" simulating "100M+ compartments." That's not what's implemented here,
and it's worth being precise about the gap rather than let the adjectives
stand in for it:

- **No GPU execution.** `cuda_kernels/hh_solver.cu` is a hand-translation of
  the Python engine's math into CUDA C. It has never been compiled or run —
  this environment has no CUDA toolchain or device. Treat it as a starting
  point that needs real on-GPU numerical-parity testing against the NumPy
  reference before it's trusted for anything.
- **No 100M-compartment path.** The multi-compartment solver is a correct,
  tested O(N) Hines tree-elimination for a single neuron's compartment tree
  (soma + branching dendrites). Scaling to 100M+ compartments needs, at
  minimum: a batched/GPU version of the tree solve, sparse morphology
  storage, and almost certainly a distributed (multi-GPU or multi-node)
  execution model. None of that exists yet.
- **"Zero-overhead"** isn't a real target as stated — everything here has the
  overhead of whatever it actually is (Python/NumPy call overhead on CPU).
  No profiling has been done to substantiate any performance claim.
- **The UniProt/PubMed "knowledge engine"** is a keyword-overlap document
  store (`agents/knowledge_engine.py`), not a semantic search index and not
  connected to any live UniProt/PubMed API. `utils/parameterizer.py`'s
  variant table is a small hand-seeded example (SCN8A/NaV1.6, KCND2/KV4.2,
  GRIN2A/GluN2A), not a curated electrophysiology database.

## Component map

```
agents/
  base.py              AgentType, ResearchTask/Result, NeuroAgent ABC
  knowledge_engine.py   keyword-overlap document store + query
  molecular_agent.py    UniProt accession -> HH engine parameter routing;
                         receptor-occupancy (Hill eq.) and STDP-style
                         plasticity estimators
engines/
  hodgkin_huxley_vec.py  vectorized 4-state (V,m,h,n) HH engine;
                          forward-Euler+Euler-Maruyama and RK4 integrators
  multi_compartment.py   Hines tree-elimination solver: implicit axial
                          coupling, explicit ionic currents, O(N) per step
cuda_kernels/
  hh_solver.cu           unverified CUDA translation of the HH step
utils/
  parameterizer.py       VariantParameter + a small seed table of known
                          UniProt accessions -> kinetic parameter shifts
```

## Numerical methods

**HH engine** uses the classic squid-axon parameter set (`g_Na=120`,
`g_K=36`, `g_leak=0.3 mS/cm^2`; `E_Na=50`, `E_K=-77`, `E_leak=-54.387 mV`)
and the standard NEURON `hh.mod` alpha/beta rate formulas, vectorized across
a batch dimension (one set of states per neuron/variant in a sweep). Verified
by test: stable resting potential, no spiking below threshold, repetitive
spiking above threshold, ISI variability under Euler-Maruyama voltage noise,
and reduced excitability under a halved `g_Na`.

**Multi-compartment engine** treats axial (linear, inter-compartment) current
implicitly and ionic (nonlinear, per-compartment HH) current explicitly — a
standard operator-split scheme for cable equations. The implicit linear
system is solved by tree-structured Gaussian elimination (Hines 1984):
process compartments leaf-to-root, folding each child's contribution into its
parent's equation, then back-substitute root-to-leaves. This is O(N) per
step with no matrix ever assembled, and works for arbitrary tree topologies
(unbranched chains and branching dendrites alike), not just the two
compartments shown in usage examples. Axial conductance is derived from
compartment diameter/length/resistivity under standard cable-theory
simplifications (uniform diameter, length = inter-center distance, end
effects ignored) — see `Compartment.axial_conductance_density`.

## Honest next steps, in priority order

1. Compile and numerically validate `hh_solver.cu` against the NumPy
   reference on real GPU hardware.
2. Decide whether the "agents" layer should stay a lightweight routing shim
   or grow into something backed by real literature retrieval/embeddings.
3. If large-compartment-count simulation is an actual goal, scope a
   batched/GPU tree solve and sparse morphology representation as its own
   project, informed by existing sparse-Hines GPU literature (e.g. CoreNEURON),
   rather than retrofitting the current single-neuron Python solver.
