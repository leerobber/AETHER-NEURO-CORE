"""Coordinates molecular metadata into HH parameter sweeps.

`MolecularNeuroAgent` is a SUB-level agent that fans a task out to four
micro-agent roles — `ion_channel_modeler`, `receptor_binding_predictor`,
`synaptic_plasticity_analyzer`, and `knowledge_engine` — and uses the
ion-channel-modeler role to turn a list of UniProt accessions into a batch of
`HighThroughputHHEngine` conductance/kinetic-shift parameters via
`aether_neuro_core.utils.parameterizer`.

The receptor-binding and synaptic-plasticity roles are represented here as
simple, honestly-scoped estimators (NMDA occupancy from a Hill equation;
Hebbian-style plasticity delta from a spike-timing window) rather than as
literature-scale predictive models — swapping in a real trained model behind
the same method signature is the intended extension point.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aether_neuro_core.agents.base import AgentType, NeuroAgent, ResearchResult, ResearchTask
from aether_neuro_core.agents.knowledge_engine import KnowledgeEngine
from aether_neuro_core.engines.hodgkin_huxley_vec import HighThroughputHHEngine
from aether_neuro_core.utils.parameterizer import VariantParameter, lookup_variant


def receptor_binding_predictor(concentration_uM: float, kd_uM: float = 5.0, hill_n: float = 1.0) -> float:
    """Fractional receptor occupancy via the Hill equation. Returns a value in [0, 1]."""
    if concentration_uM < 0:
        raise ValueError("concentration_uM must be non-negative")
    c_n = concentration_uM**hill_n
    return c_n / (kd_uM**hill_n + c_n)


def synaptic_plasticity_analyzer(pre_spike_ms: float, post_spike_ms: float, tau_ms: float = 20.0) -> float:
    """Simple pairwise STDP weight-change estimate for one pre/post spike pair.

    Positive when post follows pre (potentiation), negative when pre follows
    post (depression), decaying exponentially with |delta_t| / tau_ms.
    """
    delta_t = post_spike_ms - pre_spike_ms
    sign = 1.0 if delta_t >= 0 else -1.0
    return sign * float(np.exp(-abs(delta_t) / tau_ms))


class MolecularNeuroAgent(NeuroAgent):
    """Routes molecular-variant metadata into HH engine parameters for simulation."""

    def __init__(self, name: str = "molecular_agent", knowledge_engine: Optional[KnowledgeEngine] = None) -> None:
        super().__init__(name=name, agent_type=AgentType.SUB)
        self.knowledge_engine = knowledge_engine or KnowledgeEngine()
        self.knowledge_base["micro_agents"] = [
            "ion_channel_modeler",
            "receptor_binding_predictor",
            "synaptic_plasticity_analyzer",
            "knowledge_engine",
        ]

    async def execute(self, task: ResearchTask) -> ResearchResult:
        action = task.parameters.get("action", "build_engine")
        try:
            if action == "build_engine":
                accessions: List[str] = task.parameters["uniprot_accessions"]
                dt = task.parameters.get("dt", 0.01)
                engine = self.ion_channel_modeler(accessions, dt=dt)
                return self._result(
                    task,
                    success=True,
                    payload={
                        "batch_size": engine.batch_size,
                        "accessions": accessions,
                    },
                )
            elif action == "receptor_binding":
                occ = receptor_binding_predictor(
                    task.parameters["concentration_uM"],
                    kd_uM=task.parameters.get("kd_uM", 5.0),
                    hill_n=task.parameters.get("hill_n", 1.0),
                )
                return self._result(task, success=True, payload={"occupancy": occ})
            elif action == "synaptic_plasticity":
                delta_w = synaptic_plasticity_analyzer(
                    task.parameters["pre_spike_ms"],
                    task.parameters["post_spike_ms"],
                    tau_ms=task.parameters.get("tau_ms", 20.0),
                )
                return self._result(task, success=True, payload={"delta_weight": delta_w})
            else:
                return self._result(task, success=False, error=f"Unknown action: {action}")
        except Exception as exc:  # noqa: BLE001
            return self._result(task, success=False, error=str(exc))

    def resolve_variants(self, uniprot_accessions: List[str]) -> List[VariantParameter]:
        """Look up known kinetic parameters for a list of UniProt accessions.

        Unknown accessions fall back to a wild-type-like default (no shift,
        unit conductance scale) rather than raising, so a sweep can mix known
        and not-yet-annotated variants.
        """
        variants = []
        for accession in uniprot_accessions:
            variant = lookup_variant(accession)
            if variant is None:
                variant = VariantParameter(
                    uniprot_accession=accession,
                    gene_symbol="UNKNOWN",
                    channel_name="UNKNOWN",
                    description=f"No annotation found for {accession}; using wild-type defaults.",
                )
            variants.append(variant)
        return variants

    def ion_channel_modeler(
        self,
        uniprot_accessions: List[str],
        base_g_na: float = 120.0,
        base_g_k: float = 36.0,
        dt: float = 0.01,
        seed: Optional[int] = None,
    ) -> HighThroughputHHEngine:
        """Build a batched HH engine, one neuron per resolved variant, with
        conductances and kinetic shifts derived from each variant's parameters.
        """
        variants = self.resolve_variants(uniprot_accessions)
        g_na = np.array([base_g_na * v.g_na_scale for v in variants])
        g_k = np.array([base_g_k * v.g_k_scale for v in variants])
        na_shift = np.array([v.half_activation_shift_mv for v in variants])

        return HighThroughputHHEngine(
            batch_size=len(variants),
            g_na=g_na,
            g_k=g_k,
            na_v_shift_mv=na_shift,
            dt=dt,
            seed=seed,
        )
