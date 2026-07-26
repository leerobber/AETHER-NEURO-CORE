import pytest

from aether_neuro_core.agents.base import AgentType, LiteratureDocument, ResearchTask
from aether_neuro_core.agents.knowledge_engine import KnowledgeEngine
from aether_neuro_core.agents.molecular_agent import (
    MolecularNeuroAgent,
    receptor_binding_predictor,
    synaptic_plasticity_analyzer,
)
from aether_neuro_core.utils.parameterizer import KNOWN_VARIANTS, lookup_variant


@pytest.mark.asyncio
async def test_knowledge_engine_ingest_and_query():
    engine = KnowledgeEngine()
    doc = LiteratureDocument(
        doc_id="doc1",
        source="uniprot",
        title="SCN8A sodium channel",
        text="NaV1.6 is the dominant sodium channel at the axon initial segment.",
    )
    ingest_task = ResearchTask(query="ingest", parameters={"action": "ingest", "document": doc})
    result = await engine.execute(ingest_task)
    assert result.success
    assert result.payload["ingested"] == "doc1"

    query_task = ResearchTask(query="sodium channel axon", parameters={"action": "query", "top_k": 3})
    query_result = await engine.execute(query_task)
    assert query_result.success
    assert len(query_result.documents) == 1
    assert query_result.documents[0].doc_id == "doc1"


@pytest.mark.asyncio
async def test_knowledge_engine_query_with_no_matches_returns_empty():
    engine = KnowledgeEngine()
    task = ResearchTask(query="completely unrelated topic xyz", parameters={"action": "query"})
    result = await engine.execute(task)
    assert result.success
    assert result.documents == []


@pytest.mark.asyncio
async def test_knowledge_engine_unknown_action_fails_gracefully():
    engine = KnowledgeEngine()
    task = ResearchTask(query="x", parameters={"action": "not_a_real_action"})
    result = await engine.execute(task)
    assert not result.success
    assert result.error is not None


def test_knowledge_engine_export_and_load_roundtrip(tmp_path):
    engine = KnowledgeEngine()
    doc = LiteratureDocument(doc_id="d1", source="pubmed", title="t", text="some text about ion channels")
    engine.ingest(doc)

    path = tmp_path / "kb.json"
    engine.export_json(path)

    restored = KnowledgeEngine()
    restored.load_json(path)
    results = restored.query("ion channels")
    assert len(results) == 1
    assert results[0].doc_id == "d1"


def test_agent_type_enum_values():
    assert AgentType.SUPER.value == "super"
    assert AgentType.SUB.value == "sub"
    assert AgentType.MICRO.value == "micro"
    assert AgentType.NANO.value == "nano"


def test_lookup_variant_known_and_unknown():
    variant = lookup_variant("Q92913")
    assert variant is not None
    assert variant.gene_symbol == "SCN8A"
    assert variant.channel_name == "NaV1.6"
    assert lookup_variant("NOT_A_REAL_ACCESSION") is None
    assert "Q9UK17" in KNOWN_VARIANTS  # KCND2 / KV4.2 seeded


@pytest.mark.asyncio
async def test_molecular_agent_build_engine_known_and_unknown_variants():
    agent = MolecularNeuroAgent()
    task = ResearchTask(
        query="build sweep",
        parameters={"action": "build_engine", "uniprot_accessions": ["Q92913", "UNKNOWN_ACC"]},
    )
    result = await agent.execute(task)
    assert result.success
    assert result.payload["batch_size"] == 2


def test_molecular_agent_ion_channel_modeler_applies_variant_scaling():
    agent = MolecularNeuroAgent()
    from aether_neuro_core.utils.parameterizer import VariantParameter, register_variant

    register_variant(
        VariantParameter(
            uniprot_accession="TEST_GAIN",
            gene_symbol="TESTG",
            channel_name="TestChannel",
            g_na_scale=2.0,
        )
    )
    engine = agent.ion_channel_modeler(["Q92913", "TEST_GAIN"], base_g_na=100.0)
    assert engine.g_na[0] == pytest.approx(100.0)
    assert engine.g_na[1] == pytest.approx(200.0)


def test_receptor_binding_predictor_bounds():
    assert receptor_binding_predictor(0.0) == 0.0
    assert 0.0 < receptor_binding_predictor(5.0, kd_uM=5.0) < 1.0
    assert receptor_binding_predictor(1e6, kd_uM=5.0) > 0.99

    with pytest.raises(ValueError):
        receptor_binding_predictor(-1.0)


def test_synaptic_plasticity_analyzer_sign_convention():
    potentiation = synaptic_plasticity_analyzer(pre_spike_ms=0.0, post_spike_ms=5.0)
    depression = synaptic_plasticity_analyzer(pre_spike_ms=5.0, post_spike_ms=0.0)
    assert potentiation > 0
    assert depression < 0


@pytest.mark.asyncio
async def test_molecular_agent_receptor_and_plasticity_actions():
    agent = MolecularNeuroAgent()

    occ_task = ResearchTask(
        query="occupancy", parameters={"action": "receptor_binding", "concentration_uM": 10.0, "kd_uM": 5.0}
    )
    occ_result = await agent.execute(occ_task)
    assert occ_result.success
    assert 0.0 < occ_result.payload["occupancy"] < 1.0

    plasticity_task = ResearchTask(
        query="stdp", parameters={"action": "synaptic_plasticity", "pre_spike_ms": 0.0, "post_spike_ms": 10.0}
    )
    plasticity_result = await agent.execute(plasticity_task)
    assert plasticity_result.success
    assert plasticity_result.payload["delta_weight"] > 0
