"""Maps molecular metadata (UniProt accessions) to Hodgkin-Huxley kinetic parameters.

The values here are illustrative, order-of-magnitude-plausible shifts for a small
set of well-known ion-channel variants — not a substitute for a curated
electrophysiology database. Treat `KNOWN_VARIANTS` as a seed table meant to be
extended, not as ground truth for real biological claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class VariantParameter:
    """Perturbations a single genetic variant applies to HH channel kinetics.

    `half_activation_shift_mv` shifts the voltage-dependent activation midpoint
    (Delta V_1/2); `g_na_scale` / `g_k_scale` scale the corresponding maximal
    conductances relative to wild type (1.0 = no change).
    """

    uniprot_accession: str
    gene_symbol: str
    channel_name: str
    half_activation_shift_mv: float = 0.0
    g_na_scale: float = 1.0
    g_k_scale: float = 1.0
    description: str = ""
    extra: Dict[str, float] = field(default_factory=dict)


# Seed table of well-characterized ion-channel / receptor subunits relevant to
# Layer 5 pyramidal neuron soma/dendrite modeling.
KNOWN_VARIANTS: Dict[str, VariantParameter] = {
    "Q92913": VariantParameter(
        uniprot_accession="Q92913",
        gene_symbol="SCN8A",
        channel_name="NaV1.6",
        half_activation_shift_mv=0.0,
        g_na_scale=1.0,
        description="Wild-type SCN8A / NaV1.6, dominant somatic/axonal sodium channel.",
    ),
    "Q9UK17": VariantParameter(
        uniprot_accession="Q9UK17",
        gene_symbol="KCND2",
        channel_name="KV4.2",
        g_k_scale=1.0,
        description="Wild-type KCND2 / KV4.2, A-type dendritic potassium channel.",
    ),
    "Q12879": VariantParameter(
        uniprot_accession="Q12879",
        gene_symbol="GRIN2A",
        channel_name="GluN2A",
        description="Wild-type GRIN2A / GluN2A, NMDA receptor subunit.",
        extra={"nmda_tau_decay_ms": 55.0},
    ),
}


def lookup_variant(uniprot_accession: str) -> Optional[VariantParameter]:
    """Return the known VariantParameter for a UniProt accession, or None."""
    return KNOWN_VARIANTS.get(uniprot_accession)


def register_variant(variant: VariantParameter) -> None:
    """Add or override a variant entry in the known-variants table."""
    KNOWN_VARIANTS[variant.uniprot_accession] = variant
