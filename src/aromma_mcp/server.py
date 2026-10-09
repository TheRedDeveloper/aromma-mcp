"""
AROMMA FastMCP Server

Exposes neural odor prediction with universal ratio syntax,
recommender with database toggling, and ratio-aware scent modulation.
"""

from typing import List, Optional, Dict, Any
from fastmcp import FastMCP

from aromma_mcp.model import get_predictor
from aromma_mcp.recommender import get_recommender
from aromma_mcp.vocab import get_vocabulary

mcp = FastMCP(
    "AROMMA Odor MCP Server",
    instructions="AROMMA MCP Server provides neural odor prediction with universal ratio syntax, persistent LanceDB dual-table caching, and direct scent modulation.",
)


@mcp.tool()
def predict_odor(
    smiles: str,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Predict olfactory descriptors for a single molecule or mixture with optional ratios.
    Uses Neural Synergy-Aware Mixture Interpolation for multi-component blends.

    Syntax examples:
      - 'CCO' (pure single molecule)
      - 'CCN:0.7;CCO:0.3' (ratio mixture, normalized to 70% CCN and 30% CCO)
      - 'CCN:1;CCO:2;CC(=O)O:1' (3-part blend, normalized to 25% CCN, 50% CCO, 25% Acetic Acid)
      - 'CCN;CCO' (omitted ratios default to 1:1)

    Args:
        smiles: Molecule SMILES or semicolon-separated ratio expression.
        threshold: Classification probability threshold (default 0.5).
        top_k: Maximum number of predicted descriptors to return.

    Returns:
        Dictionary with components, normalized weights, predicted descriptors, and probabilities.
    """
    predictor = get_predictor()
    return predictor.predict(
        smiles=smiles,
        threshold=threshold,
        top_k=top_k,
    )


@mcp.tool()
def recommend_molecules(
    target_descriptors: List[str],
    exclude_descriptors: Optional[List[str]] = None,
    additional_smiles: Optional[List[str]] = None,
    include_database: bool = True,
    min_half_life_hours: Optional[float] = None,
    max_half_life_hours: Optional[float] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Find single molecules matching target scent descriptors while excluding unwanted notes
    and optionally filtering by evaporation persistence (half-life in hours).

    Args:
        target_descriptors: Desired scents (e.g. ['citrus', 'floral']).
        exclude_descriptors: Scents to strictly avoid (e.g. ['fishy', 'sulfur']).
        additional_smiles: Optional additional user molecules to include in search space.
        include_database: If True (default), searches across all 4,814+ verified database molecules.
                          If False, searches exclusively within additional_smiles (strict inventory mode).
        min_half_life_hours: Minimum evaporation half-life in hours (e.g. 4.0 for heart/base notes).
        max_half_life_hours: Maximum evaporation half-life in hours (e.g. 1.0 for fleeting top notes).
        top_k: Maximum number of candidate molecules to return (default 5).

    Returns:
        Ranked list of matching single molecules with their complete descriptor profiles and half-lives.
    """
    recommender = get_recommender()
    return recommender.recommend(
        target_descriptors=target_descriptors,
        exclude_descriptors=exclude_descriptors,
        additional_smiles=additional_smiles,
        include_database=include_database,
        min_half_life_hours=min_half_life_hours,
        max_half_life_hours=max_half_life_hours,
        top_k=top_k,
    )


@mcp.tool()
def modulate_scent(
    base: str,
    reduce_descriptors: List[str],
    boost_descriptors: Optional[List[str]] = None,
    additional_smiles: Optional[List[str]] = None,
    include_database: bool = True,
    min_half_life_hours: Optional[float] = None,
    max_half_life_hours: Optional[float] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Find what single candidate molecule from the database or inventory best fills the remaining
    budget of `base` to suppress unwanted notes (e.g. 'fishy') and optionally accentuate desired boost notes,
    with optional evaporation persistence filtering.

    Base fraction rules:
      - The `base` must be an unnormalized formula leaving an additive budget (sum of fractions < 1.0).
        For example:
          - 'CCN:0.8' (80% base, leaving 20% budget for candidate additives)
          - 'CCN:0.7;CCO:0.1' (80% total base blend, leaving 20% budget for additives)
      - If `base` sums to 1.0 or more (e.g. 'CCN' with no ratio), an error is returned.

    Args:
        base: Unnormalized formula expression leaving an additive budget (sum < 1.0).
        reduce_descriptors: Notes to suppress (e.g. ['fishy']).
        boost_descriptors: Optional notes to accentuate (e.g. ['citrus', 'sweet']).
        additional_smiles: Additional user molecules to consider as candidate additives.
        include_database: Whether to include the database molecules as candidate additives (default True).
        min_half_life_hours: Minimum half-life in hours for candidate additives (e.g. 4.0 for heart/base persistence).
        max_half_life_hours: Maximum half-life in hours for candidate additives (e.g. 1.0 for top note flash).
        top_k: Number of candidate additives to return (default 5).

    Returns:
        Ranked list of additives showing before/after simulated probabilities on the combined scent.
    """
    recommender = get_recommender()
    return recommender.modulate_scent(
        base=base,
        reduce_descriptors=reduce_descriptors,
        boost_descriptors=boost_descriptors,
        additional_smiles=additional_smiles,
        include_database=include_database,
        min_half_life_hours=min_half_life_hours,
        max_half_life_hours=max_half_life_hours,
        top_k=top_k,
    )


@mcp.tool()
def list_descriptors(search: Optional[str] = None) -> Dict[str, Any]:
    """
    List or search available olfactory descriptors in the unified 152-descriptor AROMMA vocabulary.

    Args:
        search: Optional substring to filter descriptors (e.g. 'fruit', 'woody', 'fish').

    Returns:
        List of matching descriptor names and the total count.
    """
    vocab = get_vocabulary()
    if search:
        results = vocab.search(search)
    else:
        results = vocab.labels

    return {
        "descriptors": results,
        "count": len(results),
        "total_vocabulary_size": len(vocab),
    }


@mcp.tool()
def balance_mixture(
    ingredients: List[str],
    base: Optional[str] = None,
    target_notes: Optional[Dict[str, float]] = None,
    suppress_notes: Optional[List[str]] = None,
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Optimizes ratio weights across candidate ingredients (and an optional fixed base)
    to match target scent notes and suppress unwanted notes.

    Supports compound accords as ingredients (e.g. 'limonene:0.5;citral:0.5').

    Enforcing Minimum Ingredient Bounds (Anti-Zeroing Pattern):
      If you want an ingredient to have a guaranteed minimum presence (e.g., ensuring a Floral
      accord is present at least 15%), specify that minimum fraction in `base`
      (e.g., base='floral_accord:0.15;other_locked:0.10'). The optimizer will strictly guarantee
      that 15% presence, and dynamically distribute the remaining 75% budget across candidate ingredients.

    Base fraction rules:
      - If `base` is provided (e.g. 'CCN:0.2'), its fraction is fixed and must sum to < 1.0.
      - If `base` sums to >= 1.0 (e.g. 'CCN' with no ratio), an error is returned.

    Args:
        ingredients: List of candidate ingredients (SMILES or ratio expressions).
        base: Optional unnormalized base expression (e.g. 'CCN:0.2'). Total fraction must be < 1.0.
        target_notes: Dictionary of desired descriptors and target probabilities (e.g. {'citrus': 0.8}).
        suppress_notes: List of unwanted notes to minimize (e.g. ['fishy']).
        threshold: Olfactory detection threshold for reporting active descriptors (default 0.3).

    Returns:
        Optimal ratio string, ingredient breakdown percentages, and achieved scent profile.
    """
    from aromma_mcp.optimizer import get_optimizer
    optimizer = get_optimizer()
    return optimizer.optimize_balance(
        ingredients=ingredients,
        base=base,
        target_notes=target_notes,
        suppress_notes=suppress_notes,
        threshold=threshold,
    )


@mcp.tool()
def neutralize_scent(
    ingredients: List[str],
    base: Optional[str] = None,
    threshold: float = 0.3,
    over_evaporation: bool = False,
) -> Dict[str, Any]:
    """
    Finds optimal ratios across candidate ingredients to neutralize odor by maximizing
    the 'odorless' descriptor and driving all other scent notes below threshold.

    Args:
        ingredients: List of candidate neutralizing ingredients (SMILES or ratio accords).
        base: Optional unnormalized base expression representing the odor to mask (e.g. 'CCN:0.3').
        threshold: Perception threshold below which notes are considered neutralized (default 0.3).
        over_evaporation: If False (default), optimizes instantaneous masking at t=0.
                          If True, integrates neutralization loss across an 8-hour evaporation lifespan
                          via quadrature, ensuring persistent counter-agents prevent the foul base from breaking out.

    Returns:
        Optimal recipe, odorless probability, whether full neutralization was achieved,
        and residual notes.
    """
    from aromma_mcp.optimizer import get_optimizer
    optimizer = get_optimizer()
    return optimizer.optimize_neutralize(
        ingredients=ingredients,
        base=base,
        threshold=threshold,
        over_evaporation=over_evaporation,
    )


@mcp.tool()
def balance_perfume(
    ingredients: List[str],
    timeline_targets: Dict[float, Dict[str, float]],
    base: Optional[str] = None,
    suppress_notes: Optional[List[str]] = None,
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Multi-phase fine fragrance architecture optimizer.
    Finds ingredient ratios to satisfy evolving olfactory targets across keyframe hours
    (e.g. Opening at 0h, Heart at 2h, Drydown at 8h) while suppressing unwanted notes across all phases.

    Example timeline_targets:
      {
        0.0: {'citrus': 0.8, 'fresh': 0.6},
        2.0: {'floral': 0.8, 'rose': 0.5},
        8.0: {'woody': 0.7, 'vanilla': 0.5}
      }

    Args:
        ingredients: List of candidate ingredients (SMILES or ratio accords).
        timeline_targets: Dictionary mapping elapsed hours (float) to target descriptor probabilities.
        base: Optional unnormalized base expression (e.g. 'CCN:0.1' or locked accords). Total fraction must be < 1.0.
        suppress_notes: Notes to strictly minimize across all timeframes.
        threshold: Olfactory detection threshold for reporting active descriptors (default 0.3).

    Returns:
        Optimal formula string, ingredient percentages, and detailed breakdown of notes achieved
        at each requested timeline hour.
    """
    from aromma_mcp.optimizer import get_optimizer
    optimizer = get_optimizer()
    return optimizer.optimize_perfume_timeline(
        ingredients=ingredients,
        timeline_targets=timeline_targets,
        base=base,
        suppress_notes=suppress_notes,
        threshold=threshold,
    )


@mcp.tool()
def scent_after(
    smiles: str,
    hours: List[float],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Inspects how any perfume formula, accord, or single molecule evolves over time as components evaporate.

    Args:
        smiles: Molecule SMILES or ratio formula (e.g. 'limonene:0.6;vanillin:0.4').
        hours: List of time checkpoints in hours to evaluate (e.g. [0.0, 1.0, 4.0, 8.0]).
        threshold: Olfactory perception threshold for reporting active descriptors (default 0.3).

    Returns:
        Timeline trajectory showing shifted liquid composition, remaining liquid mass percentage,
        and predicted scent descriptors at each requested hour.
    """
    from aromma_mcp.model import get_predictor
    from aromma_mcp.utils.chemistry import parse_ratio_expression, evolve_composition_at_time, format_ratio_expression_from_dict
    predictor = get_predictor()

    parsed = parse_ratio_expression(smiles)
    if not parsed:
        raise ValueError(f"Invalid SMILES or ratio expression: '{smiles}'")

    comp_smis, comp_weights = parsed
    timeline_results = []

    for hr in sorted(hours):
        h_float = float(hr)
        ev_mols, ev_w, rem_mass = evolve_composition_at_time(comp_smis, comp_weights, h_float)
        recipe_dict = {s: w for s, w in zip(ev_mols, ev_w)}
        ev_str = format_ratio_expression_from_dict(recipe_dict)

        pred_res = predictor.predict(ev_str, threshold=threshold)
        timeline_results.append({
            "time_hours": h_float,
            "remaining_liquid_mass_percentage": round(rem_mass * 100.0, 2),
            "evolved_composition_string": ev_str,
            "predicted_descriptors": pred_res["predicted_descriptors"],
            "predictions": pred_res["predictions"][:8],
        })

    return {
        "input_smiles": smiles,
        "hours_evaluated": hours,
        "timeline": timeline_results,
    }


def main():
    """Runs the FastMCP server via stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
