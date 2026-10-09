import logging
from typing import List, Dict, Any, Optional, Set
from aromma_mcp.db import get_db
from aromma_mcp.vocab import get_vocabulary
from aromma_mcp.utils.chemistry import (
    canonicalize_single_smiles,
    parse_ratio_expression,
)

logger = logging.getLogger("aromma_mcp.recommender")


class MoleculeRecommender:
    """
    Sub-millisecond single molecule recommendation and scent modulation engine.
    Supports:
    - include_database: bool (defaults to True)
    - additional_smiles: Optional list of additional user candidates
    - Arbitrary mixture ratios in modulate_scent and predict_odor
    """

    def __init__(self):
        self.db = get_db()
        self.vocab = get_vocabulary()
        logger.info(f"MoleculeRecommender initialized with {self.db.count()} molecules in LanceDB.")

    def recommend(
        self,
        target_descriptors: List[str],
        exclude_descriptors: Optional[List[str]] = None,
        additional_smiles: Optional[List[str]] = None,
        include_database: bool = True,
        min_half_life_hours: Optional[float] = None,
        max_half_life_hours: Optional[float] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Recommends single molecules matching target scents while excluding unwanted notes
        and optionally filtering by evaporation half-life persistence.

        Args:
            target_descriptors: Desired scents (e.g. ['citrus', 'floral']).
            exclude_descriptors: Scents to avoid (e.g. ['fishy', 'sulfur']).
            additional_smiles: User-supplied molecules to include in search space.
            include_database: Whether to include the precomputed 4,814+ database molecules (default True).
                              Set to False to search strictly within additional_smiles.
            min_half_life_hours: Minimum evaporation half-life in hours (e.g. 4.0 for heart/base persistence).
            max_half_life_hours: Maximum evaporation half-life in hours (e.g. 1.0 for fleeting top notes).
            top_k: Max molecules to return.
        """
        from aromma_mcp.model import get_predictor
        from aromma_mcp.utils.chemistry import estimate_molecule_half_life
        predictor = get_predictor()

        targets: Set[str] = {t.strip().lower() for t in target_descriptors if t.strip()}
        excludes: Set[str] = (
            {e.strip().lower() for e in exclude_descriptors if e.strip()}
            if exclude_descriptors
            else set()
        )

        # Pre-seed any novel additional_smiles into LanceDB 'molecules'
        if additional_smiles:
            for s in additional_smiles:
                parsed = parse_ratio_expression(s)
                if parsed:
                    for comp_smi in parsed[0]:
                        if not self.db.get_molecule(comp_smi):
                            predictor.get_or_predict_single(comp_smi)

        # Fetch candidate molecules according to flags
        searchable_entries = self.db.get_searchable_molecules(
            additional_smiles=additional_smiles,
            include_database=include_database,
        )

        candidates = []
        for entry in searchable_entries:
            smi = entry["smiles"]
            mol_desc = set(entry.get("descriptors", []))
            if excludes and (mol_desc & excludes):
                continue

            # Persistence filter
            th = estimate_molecule_half_life(smi)
            if min_half_life_hours is not None and th < min_half_life_hours:
                continue
            if max_half_life_hours is not None and th > max_half_life_hours:
                continue

            matched = mol_desc & targets
            match_count = len(matched)
            if match_count > 0:
                candidates.append({
                    "smiles": smi,
                    "descriptors": entry.get("descriptors", []),
                    "half_life_hours": th,
                    "matched_descriptors": sorted(list(matched)),
                    "match_count": match_count,
                    "extra_descriptors_count": len(mol_desc) - match_count,
                })

        candidates.sort(
            key=lambda c: (-c["match_count"], c["extra_descriptors_count"])
        )

        results = candidates[:top_k]

        return {
            "query": {
                "target_descriptors": sorted(list(targets)),
                "exclude_descriptors": sorted(list(excludes)),
                "include_database": include_database,
                "min_half_life_hours": min_half_life_hours,
                "max_half_life_hours": max_half_life_hours,
                "additional_smiles_count": len(additional_smiles) if additional_smiles else 0,
                "candidate_pool_size": len(searchable_entries),
            },
            "total_matches": len(candidates),
            "recommendations": results,
        }

    def modulate_scent(
        self,
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
        Determines what single candidate molecule from the database or inventory best fills
        the remaining fraction budget of `base` to reduce unwanted notes (e.g., make it less 'fishy')
        and optionally boost desired notes, with optional evaporation persistence filtering.

        Args:
            base: Unnormalized formula expression leaving an additive budget (total fraction < 1.0).
                  Examples:
                    - 'CCN:0.8' (80% CCN base, leaving 20% budget for candidate additives)
                    - 'CCN:0.7;CCO:0.1' (70% CCN, 10% CCO base, leaving 20% budget for additives)
            reduce_descriptors: Notes to suppress (e.g. ['fishy']).
            boost_descriptors: Optional notes to accentuate (e.g. ['citrus', 'sweet']).
            additional_smiles: Additional user molecules to consider as candidate additives.
            include_database: Whether to include the database molecules as candidate additives (default True).
            min_half_life_hours: Minimum half-life in hours for candidate additives (e.g. 4.0 for heart/base persistence).
            max_half_life_hours: Maximum half-life in hours for candidate additives (e.g. 1.0 for top note flash).
            top_k: Number of candidate additives to return (default 5).
        """
        from aromma_mcp.model import get_predictor
        from aromma_mcp.utils.chemistry import parse_unnormalized_expression, estimate_molecule_half_life
        predictor = get_predictor()

        parsed_base = parse_unnormalized_expression(base)
        if not parsed_base:
            raise ValueError(f"Invalid base expression: '{base}'")

        base_smiles_list, raw_weights = parsed_base
        total_base_weight = sum(raw_weights)

        if total_base_weight >= 1.0 - 1e-6:
            raise ValueError(
                f"Base total fraction is {round(total_base_weight, 4):.4f} (>= 1.0). "
                "To modulate, specify an unnormalized base fraction < 1.0 (e.g. 'CCN:0.8') "
                "to leave an additive budget (e.g. 0.2) for candidate molecules."
            )

        w_cand = 1.0 - total_base_weight  # Additive candidate budget fraction

        reduce_targets = [r.strip().lower() for r in reduce_descriptors if r.strip()]
        boost_targets = (
            [b.strip().lower() for b in boost_descriptors if b.strip()]
            if boost_descriptors
            else []
        )

        # Baseline evaluation of the pure base normalized
        base_normalized_smis = base_smiles_list
        base_normalized_weights = [round(w / total_base_weight, 4) for w in raw_weights]
        base_norm_str = ";".join([f"{s}:{w}" for s, w in zip(base_normalized_smis, base_normalized_weights)])
        base_pred = predictor.predict(base_norm_str, threshold=0.01)
        base_probs = {p["descriptor"]: p["probability"] for p in base_pred["predictions"]}

        unwanted_indices = [
            self.vocab.labels.index(r) for r in reduce_targets if r in self.vocab.labels
        ]
        boost_indices = [
            self.vocab.labels.index(b) for b in boost_targets if b in self.vocab.labels
        ]

        # Pre-seed additional_smiles if provided
        if additional_smiles:
            for s in additional_smiles:
                p_item = parse_ratio_expression(s)
                if p_item:
                    for comp in p_item[0]:
                        if not self.db.get_molecule(comp):
                            predictor.get_or_predict_single(comp)

        searchable_entries = self.db.get_searchable_molecules(
            additional_smiles=additional_smiles,
            include_database=include_database,
        )

        current_components = set(base_smiles_list)
        candidate_molecules = []
        for entry in searchable_entries:
            smi = entry["smiles"]
            if smi in current_components:
                continue
            entry_desc = set(entry.get("descriptors", []))
            if any(r in entry_desc for r in reduce_targets):
                continue

            # Persistence filter
            th = estimate_molecule_half_life(smi)
            if min_half_life_hours is not None and th < min_half_life_hours:
                continue
            if max_half_life_hours is not None and th > max_half_life_hours:
                continue

            boost_overlap = len(entry_desc & set(boost_targets)) if boost_targets else 0
            candidate_molecules.append((smi, boost_overlap, entry.get("descriptors", []), th))

        # No artificial caps: test all eligible candidates in the database/selection
        candidate_pool = candidate_molecules

        if not candidate_pool:
            return {
                "base": base,
                "additive_fraction": round(w_cand, 4),
                "reduce_descriptors": reduce_targets,
                "boost_descriptors": boost_targets,
                "recommendations": [],
                "message": "No suitable candidate molecules found matching the criteria.",
            }

        # High-throughput batch evaluation:
        # Candidate additive C is blended at exactly fraction w_cand with base constituents at raw_weights.
        # Notice that sum(raw_weights) + w_cand == 1.0.
        import torch

        # Ensure all constituent single molecules and candidate single molecules are cached
        all_unique_constituents = list(base_smiles_list)
        candidate_smiles_list = [c[0] for c in candidate_pool]
        for c_smi in candidate_smiles_list:
            if c_smi not in all_unique_constituents:
                all_unique_constituents.append(c_smi)

        predictor.ensure_molecules_and_interactions_cached(all_unique_constituents)

        # Base constituents
        p_base_components = [
            torch.tensor(predictor.get_or_predict_single(smi))
            for smi in base_smiles_list
        ]
        N_base = len(base_smiles_list)

        # Baseline linear contribution from base
        p_base_linear = sum(w * p for w, p in zip(raw_weights, p_base_components))
        # Internal base synergies
        p_base_internal_synergy = torch.zeros_like(p_base_components[0])
        for i in range(N_base):
            for j in range(i + 1, N_base):
                w_ij = raw_weights[i] * raw_weights[j]
                if w_ij > 0:
                    delta = torch.tensor(predictor.get_or_predict_interaction(base_smiles_list[i], base_smiles_list[j]))
                    p_base_internal_synergy += 4.0 * w_ij * delta

        # Now evaluate each candidate additive C blended at fraction w_cand
        ranked = []
        for cand_smi, _, cand_desc, cand_th in candidate_pool:
            p_cand_vec = torch.tensor(predictor.get_or_predict_single(cand_smi))

            # Cross synergy between candidate C and each base constituent:
            # sum_k 4 * raw_weights[k] * w_cand * Delta(k, C)
            cross_synergy = torch.zeros_like(p_cand_vec)
            for k in range(N_base):
                delta_k_cand = torch.tensor(predictor.get_or_predict_interaction(base_smiles_list[k], cand_smi))
                cross_synergy += 4.0 * raw_weights[k] * w_cand * delta_k_cand

            # Total blended prediction
            blend_probs = torch.clamp(
                p_base_linear + (w_cand * p_cand_vec) + p_base_internal_synergy + cross_synergy,
                0.0,
                1.0,
            )

            unwanted_drop = 0.0
            unwanted_after = {}
            for idx in unwanted_indices:
                lbl = self.vocab.labels[idx]
                p_before = base_probs.get(lbl, 0.0)
                p_after = float(blend_probs[idx])
                unwanted_drop += (p_before - p_after)
                unwanted_after[lbl] = {"before": round(p_before, 3), "after": round(p_after, 3)}

            boost_gain = 0.0
            boost_after = {}
            for idx in boost_indices:
                lbl = self.vocab.labels[idx]
                p_before = base_probs.get(lbl, 0.0)
                p_after = float(blend_probs[idx])
                boost_gain += (p_after - p_before)
                boost_after[lbl] = {"before": round(p_before, 3), "after": round(p_after, 3)}

            total_score = unwanted_drop + 0.5 * boost_gain
            ranked.append({
                "additive_smiles": cand_smi,
                "additive_descriptors": cand_desc,
                "half_life_hours": cand_th,
                "unwanted_probability_change": unwanted_after,
                "boost_probability_change": boost_after,
                "reduction_score": round(total_score, 4),
            })

        ranked.sort(key=lambda x: -x["reduction_score"])

        return {
            "base": base,
            "additive_fraction": round(w_cand, 4),
            "reduce_descriptors": reduce_targets,
            "boost_descriptors": boost_targets,
            "min_half_life_hours": min_half_life_hours,
            "max_half_life_hours": max_half_life_hours,
            "include_database": include_database,
            "top_additives": ranked[:top_k],
        }


_RECOMMENDER_INSTANCE: Optional[MoleculeRecommender] = None


def get_recommender() -> MoleculeRecommender:
    global _RECOMMENDER_INSTANCE
    if _RECOMMENDER_INSTANCE is None:
        _RECOMMENDER_INSTANCE = MoleculeRecommender()
    return _RECOMMENDER_INSTANCE
