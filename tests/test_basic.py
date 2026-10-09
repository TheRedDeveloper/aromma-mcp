import pytest
from pathlib import Path
import tempfile
import torch
from aromma_mcp.utils.chemistry import (
    canonicalize_single_smiles,
    parse_ratio_expression,
    canonical_ratio_string,
    canonical_pair_key,
)
from aromma_mcp.vocab import get_vocabulary
from aromma_mcp.db import OdorDatabase
from aromma_mcp.recommender import MoleculeRecommender


def test_ratio_parsing():
    # Single molecule
    assert parse_ratio_expression("CCO") == (["CCO"], [1.0])
    # Single molecule with redundant ratio
    assert parse_ratio_expression("CN:2") == (["CN"], [1.0])
    assert canonical_ratio_string("CN:2") == "CN"

    # Implicit 1:1 mixture
    smis, weights = parse_ratio_expression("CCN;CCO")
    assert smis == ["CCN", "CCO"]
    assert weights == [0.5, 0.5]

    # Explicit custom ratios
    smis2, weights2 = parse_ratio_expression("CCN:0.5;CCO")
    assert smis2 == ["CCN", "CCO"]
    # 0.5 / 1.5 = 0.3333, 1.0 / 1.5 = 0.6667
    assert round(weights2[0], 2) == 0.33
    assert round(weights2[1], 2) == 0.67

    # 3-part blend
    smis3, weights3 = parse_ratio_expression("CCN:1;CCO:2;CC(=O)O:1")
    assert len(smis3) == 3
    assert weights3 == [0.25, 0.5, 0.25]


def test_commutative_pair_key():
    k1 = canonical_pair_key("CCO", "CCN")
    k2 = canonical_pair_key("CCN", "CCO")
    assert k1 == k2
    assert k1[0] <= k1[1]


def test_dual_table_lancedb():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_dir = Path(tmpdir) / "test_lancedb"
        db = OdorDatabase(db_path=test_db_dir)

        # Upsert single molecule
        db.upsert_molecule("CCO", ["alcoholic"], [0.8]*152, source="prediction")
        assert db.get_molecule("CCO") is not None
        assert db.get_molecule("CCO")["smiles"] == "CCO"

        # Upsert binary interaction (A < B)
        db.upsert_interaction("CCN", "CCO", [0.05]*152)
        # Check lookup order independence
        inter = db.get_interaction("CCO", "CCN")
        assert inter is not None
        assert inter["pair_key"] == canonical_pair_key("CCN", "CCO")[2]


def test_recommender_include_database_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_dir = Path(tmpdir) / "test_lancedb"
        db = OdorDatabase(db_path=test_db_dir)

        db.upsert_molecule("CCO", ["alcoholic", "fresh"], [0.8]*152, source="benchmark")
        db.upsert_molecule("CCC1=C(O)C(=O)CC1", ["caramellic", "sweet"], [0.9]*152, source="benchmark")

        recommender = MoleculeRecommender.__new__(MoleculeRecommender)
        recommender.db = db
        recommender.vocab = get_vocabulary()

        # 1. include_database=True (default): finds caramellic in DB
        res1 = recommender.recommend(
            target_descriptors=["caramellic"],
            include_database=True,
            top_k=5,
        )
        assert res1["total_matches"] >= 1
        assert any("caramellic" in r["descriptors"] for r in res1["recommendations"])

        # 2. include_database=False with additional_smiles that does not contain caramellic
        res2 = recommender.recommend(
            target_descriptors=["caramellic"],
            additional_smiles=["CCO"],
            include_database=False,
            top_k=5,
        )
        assert res2["total_matches"] == 0

        # 3. include_database=False with user's own caramellic molecule
        res3 = recommender.recommend(
            target_descriptors=["caramellic"],
            additional_smiles=["CCC1=C(O)C(=O)CC1"],
            include_database=False,
            top_k=5,
        )
        assert res3["total_matches"] == 1


def test_unnormalized_expression_parsing():
    from aromma_mcp.utils.chemistry import parse_unnormalized_expression, format_ratio_expression_from_dict

    # Single molecule with default 1.0
    smis, weights = parse_unnormalized_expression("CCN")
    assert smis == ["CCN"]
    assert weights == [1.0]

    # Explicit fraction
    smis, weights = parse_unnormalized_expression("CCN:0.4")
    assert smis == ["CCN"]
    assert weights == [0.4]

    # Multi-component without normalization
    smis, weights = parse_unnormalized_expression("CCN:0.3;CCO:0.2")
    assert smis == ["CCN", "CCO"]
    assert weights == [0.3, 0.2]
    assert sum(weights) == 0.5

    # Dict formatter
    f = format_ratio_expression_from_dict({"CCO": 0.7, "CCN": 0.3})
    assert f == "CCN:0.3;CCO:0.7"


def test_optimizer_flattening_and_validation():
    from aromma_mcp.optimizer import MixtureOptimizer

    opt = MixtureOptimizer.__new__(MixtureOptimizer)
    opt.device = "cpu"

    # 1. Base fraction >= 1.0 must error
    with pytest.raises(ValueError, match="Base total fraction is"):
        opt._parse_and_flatten_setup(
            ingredients=["CCO"],
            base="CCN:1.0",
        )

    # 2. Base fraction > 1.0 must error
    with pytest.raises(ValueError, match="Base total fraction is"):
        opt._parse_and_flatten_setup(
            ingredients=["CCO"],
            base="CCN:0.6;CCO:0.5",
        )

    # 3. Valid base fraction and compound accord ingredient
    # Base: CCN:0.2 (leaves 0.8 budget)
    # Ingredient 1: "CCO:0.5;CC(=O)O:0.5"
    # Ingredient 2: "CC(=O)OC"
    unique_smis, b_vec, C_mat, W_rem, ing_meta = opt._parse_and_flatten_setup(
        ingredients=["CCO:0.5;CC(=O)O:0.5", "CC(=O)OC"],
        base="CCN:0.2",
    )
    assert abs(W_rem - 0.8) < 1e-5
    assert len(unique_smis) == 4
    assert len(ing_meta) == 2
    # C matrix columns must sum to 1.0
    assert torch.allclose(torch.sum(C_mat, dim=0), torch.ones(2))


def test_modulate_scent_with_mixture_expression():
    from aromma_mcp.server import modulate_scent

    # 1. Base >= 1.0 must error
    with pytest.raises(ValueError, match="Base total fraction is"):
        modulate_scent(
            base="CCN:1.0",
            reduce_descriptors=["fishy"],
        )

    # 2. Base < 1.0 leaving an additive budget
    # base has 70% CCN and 10% CCO (total 0.8), leaving 0.2 (20%) for the candidate additive
    res = modulate_scent(
        base="CCN:0.7;CCO:0.1",
        reduce_descriptors=["fishy"],
        additional_smiles=["CC1=CC[C@@H](CC1)C(=C)C"],
        include_database=False,
        top_k=1,
    )
    assert res["additive_fraction"] == 0.2
    assert len(res["top_additives"]) == 1
    add = res["top_additives"][0]
    assert "fishy" in add["unwanted_probability_change"]
    assert add["unwanted_probability_change"]["fishy"]["after"] < add["unwanted_probability_change"]["fishy"]["before"]
    assert "half_life_hours" in add


def test_evaporation_half_life_physics():
    from aromma_mcp.utils.chemistry import estimate_molecule_half_life, evolve_composition_at_time

    # Gas: ethylene clamped to boundary layer floor (0.15 h)
    t_gas = estimate_molecule_half_life("C=C")
    assert t_gas == 0.15

    # Ester: ethyl butyrate
    t_ester = estimate_molecule_half_life("CCCC(=O)OCC")
    assert 0.15 <= t_ester <= 1.0

    # Terpene: Limonene
    t_limonene = estimate_molecule_half_life("CC1=CC[C@@H](CC1)C(=C)C")
    assert 0.3 <= t_limonene <= 1.5

    # Terpenol: Linalool
    t_linalool = estimate_molecule_half_life("CC(=CCCC(C)(C=C)O)C")
    assert 1.5 <= t_linalool <= 6.0

    # Fixative: Vanillin
    t_vanillin = estimate_molecule_half_life("O=Cc1ccc(O)c(OC)c1")
    assert t_vanillin > 10.0

    # Mixture evolution over time
    comp = ["CC1=CC[C@@H](CC1)C(=C)C", "O=Cc1ccc(O)c(OC)c1"]
    w0 = [0.5, 0.5]
    _, w_ev, rem = evolve_composition_at_time(comp, w0, hours=4.0)
    # Fast-evaporating Limonene drops, Vanillin dominates
    assert w_ev[1] > w_ev[0]
    assert rem < 1.0


def test_recommend_and_modulate_half_life_filters():
    from aromma_mcp.server import recommend_molecules, modulate_scent

    # Recommend with min_half_life filter
    res = recommend_molecules(
        target_descriptors=["citrus"],
        additional_smiles=["CC1=CC[C@@H](CC1)C(=C)C", "O=Cc1ccc(O)c(OC)c1"],
        include_database=False,
        max_half_life_hours=1.0, # only fast limonene passes
        top_k=5,
    )
    assert all(r["half_life_hours"] <= 1.0 for r in res["recommendations"])

    # Modulate with half-life filter
    res_mod = modulate_scent(
        base="CCN:0.8",
        reduce_descriptors=["fishy"],
        additional_smiles=["CC1=CC[C@@H](CC1)C(=C)C"],
        include_database=False,
        max_half_life_hours=1.0,
        top_k=1,
    )
    assert len(res_mod["top_additives"]) == 1
    assert res_mod["top_additives"][0]["half_life_hours"] <= 1.0


def test_scent_after_and_balance_perfume():
    from aromma_mcp.server import scent_after, balance_perfume

    # Test scent_after
    res_after = scent_after(
        smiles="CC1=CC[C@@H](CC1)C(=C)C:0.5;O=Cc1ccc(O)c(OC)c1:0.5",
        hours=[0.0, 2.0],
        threshold=0.3,
    )
    assert len(res_after["timeline"]) == 2
    assert res_after["timeline"][0]["time_hours"] == 0.0
    assert res_after["timeline"][1]["time_hours"] == 2.0
    assert res_after["timeline"][1]["remaining_liquid_mass_percentage"] < 100.0

    # Test balance_perfume
    res_perfume = balance_perfume(
        ingredients=[
            "CC1=CC[C@@H](CC1)C(=C)C", # Limonene
            "O=Cc1ccc(O)c(OC)c1",       # Vanillin
        ],
        timeline_targets={
            0.0: {"citrus": 0.6},
            4.0: {"sweet": 0.5},
        },
        base="CCN:0.1",
        suppress_notes=["fishy"],
    )
    assert "timeline_evaluation" in res_perfume
    assert len(res_perfume["timeline_evaluation"]) == 2
    assert res_perfume["mode"] == "perfume"




