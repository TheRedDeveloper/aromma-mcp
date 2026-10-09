"""
Mixture Optimization Engine for AROMMA MCP.

Provides fast, exact-gradient, and multi-start optimization over ratio simplices
to balance mixtures (matching target notes, suppressing off-notes) and neutralize odors.
Accounts for compound ingredients (accords), fixed unnormalized bases, and non-linear
neural synergy terms.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F

from aromma_mcp.model import get_predictor
from aromma_mcp.vocab import get_vocabulary
from aromma_mcp.utils.chemistry import (
    canonicalize_single_smiles,
    parse_ratio_expression,
    parse_unnormalized_expression,
    format_ratio_expression_from_dict,
    estimate_molecule_half_life,
    evolve_composition_at_time,
)

logger = logging.getLogger("aromma_mcp.optimizer")


class MixtureOptimizer:
    """Optimizes ingredient ratios to meet olfactory profiles or achieve scent neutralization."""

    def __init__(self):
        self.predictor = get_predictor()
        self.vocab = get_vocabulary()
        self.device = self.predictor.device

    def _parse_and_flatten_setup(
        self,
        ingredients: List[str],
        base: Optional[str] = None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, float, List[Dict[str, Any]]]:
        """
        Parses base and ingredients, extracts unique canonical SMILES set,
        and constructs:
          - unique_smiles: [M] unique canonical SMILES
          - b: [M] base weight vector
          - C: [M, K] ingredient composition matrix (columns sum to 1.0)
          - W_rem: float remaining budget for candidate ingredients (1.0 - sum(b))
          - ing_meta: ingredient metadata
        """
        if not ingredients:
            raise ValueError("At least one ingredient must be provided.")

        # 1. Parse base
        base_weights_map: Dict[str, float] = {}
        total_base_weight = 0.0
        if base and base.strip():
            parsed_base = parse_unnormalized_expression(base)
            if not parsed_base:
                raise ValueError(f"Invalid base expression: '{base}'")
            smis, weights = parsed_base
            for s, w in zip(smis, weights):
                csmi = canonicalize_single_smiles(s)
                if not csmi:
                    raise ValueError(f"Invalid SMILES in base: '{s}'")
                base_weights_map[csmi] = base_weights_map.get(csmi, 0.0) + w
                total_base_weight += w

            if total_base_weight >= 1.0 - 1e-6:
                raise ValueError(
                    f"Base total fraction is {round(total_base_weight, 4):.4f} (>= 1.0). "
                    "Base must leave a fraction (> 0.0) for the candidate ingredients to vary."
                )

        W_rem = 1.0 - total_base_weight

        # 2. Parse candidate ingredients
        ing_meta = []
        for idx, ing_str in enumerate(ingredients):
            parsed_ing = parse_ratio_expression(ing_str)
            if not parsed_ing:
                raise ValueError(f"Invalid ingredient expression at index {idx}: '{ing_str}'")
            smis, weights = parsed_ing
            comp_dict = {}
            for s, w in zip(smis, weights):
                csmi = canonicalize_single_smiles(s)
                if not csmi:
                    raise ValueError(f"Invalid molecule in ingredient: '{s}'")
                comp_dict[csmi] = comp_dict.get(csmi, 0.0) + w
            ing_meta.append({
                "raw": ing_str,
                "composition": comp_dict,  # sums to 1.0
            })

        K = len(ingredients)

        # 3. Collect unique molecules
        all_unique = list(base_weights_map.keys())
        for ing in ing_meta:
            for s in ing["composition"].keys():
                if s not in all_unique:
                    all_unique.append(s)

        M = len(all_unique)

        # 4. Construct base vector b and ingredient matrix C
        b_vec = torch.zeros(M, dtype=torch.float32, device=self.device)
        for i, s in enumerate(all_unique):
            b_vec[i] = base_weights_map.get(s, 0.0)

        C_mat = torch.zeros((M, K), dtype=torch.float32, device=self.device)
        for k, ing in enumerate(ing_meta):
            for s, frac in ing["composition"].items():
                row_idx = all_unique.index(s)
                C_mat[row_idx, k] = frac

        return all_unique, b_vec, C_mat, W_rem, ing_meta

    def _prepare_tensors(self, unique_smiles: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensures molecules and pairwise synergy deltas are cached in LanceDB,
        and builds:
          - P_constituents: [M, D] constituent probabilities
          - Delta_synergy: [M, M, D] pairwise synergy deltas (with delta_ii = 0)
        """
        self.predictor.ensure_molecules_and_interactions_cached(unique_smiles)
        M = len(unique_smiles)
        D = len(self.vocab.labels)

        P_mat = torch.zeros((M, D), dtype=torch.float32, device=self.device)
        for i, smi in enumerate(unique_smiles):
            probs = self.predictor.get_or_predict_single(smi)
            P_mat[i] = torch.tensor(probs, dtype=torch.float32, device=self.device)

        Delta_tensor = torch.zeros((M, M, D), dtype=torch.float32, device=self.device)
        for i in range(M):
            for j in range(i + 1, M):
                delta = self.predictor.get_or_predict_interaction(unique_smiles[i], unique_smiles[j])
                d_vec = torch.tensor(delta, dtype=torch.float32, device=self.device)
                Delta_tensor[i, j] = d_vec
                Delta_tensor[j, i] = d_vec

        return P_mat, Delta_tensor

    def _predict_from_molecule_weights(
        self,
        w: torch.Tensor,
        P_mat: torch.Tensor,
        Delta_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given batch or single molecule weight tensor w, computes predicted descriptor probabilities.
        w: [..., M]
        Returns: [..., D] clamped to [0.0, 1.0]
        """
        # Linear blend: sum_i w_i * P_i -> w @ P_mat
        linear_part = torch.matmul(w, P_mat)  # [..., D]

        # Non-linear synergy: sum_{i < j} 4 * w_i * w_j * Delta_ij
        # Notice that sum_{i, j} 2 * w_i * w_j * Delta_ij = 4 * sum_{i < j} w_i * w_j * Delta_ij
        # (since Delta_ii = 0 and Delta_ij = Delta_ji).
        # Quadratic form: w^T Delta w for each descriptor
        # Using einsum: '...i, ...j, ijd -> ...d'
        synergy_part = 2.0 * torch.einsum("...i, ...j, ijd -> ...d", w, w, Delta_tensor)

        return torch.clamp(linear_part + synergy_part, 0.0, 1.0)

    def optimize_balance(
        self,
        ingredients: List[str],
        base: Optional[str] = None,
        target_notes: Optional[Dict[str, float]] = None,
        suppress_notes: Optional[List[str]] = None,
        threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Finds optimal ingredient ratios to match target scent notes and suppress unwanted notes.
        """
        unique_smis, b_vec, C_mat, W_rem, ing_meta = self._parse_and_flatten_setup(ingredients, base)
        P_mat, Delta_tensor = self._prepare_tensors(unique_smis)
        K = len(ingredients)

        target_targets = {}
        if target_notes:
            for k_desc, v_prob in target_notes.items():
                norm_desc = k_desc.strip().lower()
                if norm_desc in self.vocab.labels:
                    target_targets[self.vocab.labels.index(norm_desc)] = float(v_prob)

        suppress_indices = []
        if suppress_notes:
            for s_desc in suppress_notes:
                norm_desc = s_desc.strip().lower()
                if norm_desc in self.vocab.labels:
                    suppress_indices.append(self.vocab.labels.index(norm_desc))

        if not target_targets and not suppress_indices:
            raise ValueError("At least one target note or suppress note must be provided.")

        def compute_loss(pred_probs: torch.Tensor) -> torch.Tensor:
            # pred_probs: [..., D]
            loss = torch.zeros(pred_probs.shape[:-1], device=self.device)
            for idx_d, target_val in target_targets.items():
                loss += (pred_probs[..., idx_d] - target_val) ** 2
            for idx_s in suppress_indices:
                # Strong quadratic penalty to suppress unwanted notes
                loss += 3.0 * (pred_probs[..., idx_s] ** 2)
            return loss

        # Solve for best alpha
        best_alpha = self._solve_optimal_alpha(
            K=K,
            W_rem=W_rem,
            b_vec=b_vec,
            C_mat=C_mat,
            P_mat=P_mat,
            Delta_tensor=Delta_tensor,
            loss_fn=compute_loss,
        )

        return self._format_result(
            unique_smis=unique_smis,
            b_vec=b_vec,
            C_mat=C_mat,
            best_alpha=best_alpha,
            ing_meta=ing_meta,
            P_mat=P_mat,
            Delta_tensor=Delta_tensor,
            threshold=threshold,
            target_targets=target_targets,
            suppress_indices=suppress_indices,
            mode="balance",
        )

    def optimize_neutralize(
        self,
        ingredients: List[str],
        base: Optional[str] = None,
        threshold: float = 0.3,
        over_evaporation: bool = False,
    ) -> Dict[str, Any]:
        """
        Finds optimal ratios to neutralize scent by maximizing the 'odorless' descriptor
        and minimizing all other descriptor energies below threshold.

        If over_evaporation is True, integrates the neutralization loss across an 8-hour
        evaporation lifespan via quadrature, preventing volatile maskers from abandoning
        a persistent foul base.
        """
        unique_smis, b_vec, C_mat, W_rem, ing_meta = self._parse_and_flatten_setup(ingredients, base)
        P_mat, Delta_tensor = self._prepare_tensors(unique_smis)
        K = len(ingredients)

        odorless_idx = self.vocab.labels.index("odorless") if "odorless" in self.vocab.labels else -1

        # Half-lives for each constituent molecule
        half_lives = torch.tensor(
            [estimate_molecule_half_life(smi) for smi in unique_smis],
            dtype=torch.float32,
            device=self.device,
        )  # [M]

        def compute_single_time_loss(pred_probs: torch.Tensor) -> torch.Tensor:
            # Penalize any scent descriptor that is active (excluding odorless)
            mask = torch.ones(len(self.vocab.labels), dtype=torch.bool, device=self.device)
            if odorless_idx >= 0:
                mask[odorless_idx] = False

            scent_probs = pred_probs[..., mask]
            scent_penalty = torch.sum(scent_probs ** 2, dim=-1) + 2.0 * torch.sum(F.relu(scent_probs - threshold) ** 2, dim=-1)

            loss = scent_penalty
            if odorless_idx >= 0:
                loss -= 2.0 * pred_probs[..., odorless_idx]
            return loss

        if not over_evaporation:
            # Instantaneous loss at t=0
            def loss_fn(pred_probs: torch.Tensor) -> torch.Tensor:
                return compute_single_time_loss(pred_probs)

            best_alpha = self._solve_optimal_alpha(
                K=K,
                W_rem=W_rem,
                b_vec=b_vec,
                C_mat=C_mat,
                P_mat=P_mat,
                Delta_tensor=Delta_tensor,
                loss_fn=loss_fn,
            )
        else:
            # Gauss-Legendre quadrature points over [0, 8 hours]
            # Nodes and weights for 5-point quadrature scaled to [0, 8]
            t_nodes = torch.tensor([0.375, 1.846, 4.0, 6.154, 7.625], dtype=torch.float32, device=self.device)
            w_nodes = torch.tensor([0.947, 1.914, 2.276, 1.914, 0.947], dtype=torch.float32, device=self.device)

            def loss_fn_temporal(alphas: torch.Tensor) -> torch.Tensor:
                # alphas: [Batch, K]
                # w_0 = b + C @ alpha -> [Batch, M]
                w_0 = b_vec.unsqueeze(0) + torch.matmul(alphas, C_mat.t())  # [Batch, M]

                total_temporal_loss = torch.zeros(alphas.shape[0], device=self.device)
                for t_val, q_wt in zip(t_nodes, w_nodes):
                    # Decay: mass_i(t) = w_0,i * 2^(-t / t_half,i)
                    decay = 2.0 ** (-t_val / half_lives)  # [M]
                    m_t = w_0 * decay.unsqueeze(0)        # [Batch, M]
                    total_m = torch.sum(m_t, dim=-1, keepdim=True) + 1e-8
                    w_t = m_t / total_m                   # [Batch, M]

                    preds_t = self._predict_from_molecule_weights(w_t, P_mat, Delta_tensor)
                    step_loss = compute_single_time_loss(preds_t)
                    total_temporal_loss += q_wt * step_loss

                return total_temporal_loss

            best_alpha = self._solve_optimal_alpha_direct_weights(
                K=K,
                W_rem=W_rem,
                loss_fn=loss_fn_temporal,
            )

        res = self._format_result(
            unique_smis=unique_smis,
            b_vec=b_vec,
            C_mat=C_mat,
            best_alpha=best_alpha,
            ing_meta=ing_meta,
            P_mat=P_mat,
            Delta_tensor=Delta_tensor,
            threshold=threshold,
            target_targets={},
            suppress_indices=[],
            mode="neutralize",
        )
        res["over_evaporation"] = over_evaporation
        return res

    def optimize_perfume_timeline(
        self,
        ingredients: List[str],
        timeline_targets: Dict[float, Dict[str, float]],
        base: Optional[str] = None,
        suppress_notes: Optional[List[str]] = None,
        threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Multi-phase fragrance architecture optimizer.
        Optimizes ingredient ratios to satisfy keyframe timeline targets across hours
        (e.g. 0h: opening, 2h: heart, 8h: drydown) while suppressing unwanted notes.
        """
        if not timeline_targets:
            raise ValueError("timeline_targets dictionary must contain at least one keyframe (e.g. {0.0: {'citrus': 0.8}}).")

        unique_smis, b_vec, C_mat, W_rem, ing_meta = self._parse_and_flatten_setup(ingredients, base)
        P_mat, Delta_tensor = self._prepare_tensors(unique_smis)
        K = len(ingredients)

        half_lives = torch.tensor(
            [estimate_molecule_half_life(smi) for smi in unique_smis],
            dtype=torch.float32,
            device=self.device,
        )

        suppress_indices = []
        if suppress_notes:
            for s_desc in suppress_notes:
                norm_desc = s_desc.strip().lower()
                if norm_desc in self.vocab.labels:
                    suppress_indices.append(self.vocab.labels.index(norm_desc))

        # Parse keyframes
        parsed_keyframes = []
        for t_hr, targets_dict in timeline_targets.items():
            t_float = float(t_hr)
            t_dict = {}
            for k_desc, v_prob in targets_dict.items():
                norm_desc = k_desc.strip().lower()
                if norm_desc in self.vocab.labels:
                    t_dict[self.vocab.labels.index(norm_desc)] = float(v_prob)
            parsed_keyframes.append((t_float, t_dict))

        def timeline_loss_fn(alphas: torch.Tensor) -> torch.Tensor:
            # alphas: [Batch, K]
            w_0 = b_vec.unsqueeze(0) + torch.matmul(alphas, C_mat.t())  # [Batch, M]
            batch_loss = torch.zeros(alphas.shape[0], device=self.device)

            for t_float, t_dict in parsed_keyframes:
                decay = 2.0 ** (-t_float / half_lives)
                m_t = w_0 * decay.unsqueeze(0)
                total_m = torch.sum(m_t, dim=-1, keepdim=True) + 1e-8
                w_t = m_t / total_m

                preds_t = self._predict_from_molecule_weights(w_t, P_mat, Delta_tensor)

                for idx_d, target_val in t_dict.items():
                    batch_loss += (preds_t[..., idx_d] - target_val) ** 2

                for idx_s in suppress_indices:
                    batch_loss += 3.0 * (preds_t[..., idx_s] ** 2)

            return batch_loss

        best_alpha = self._solve_optimal_alpha_direct_weights(
            K=K,
            W_rem=W_rem,
            loss_fn=timeline_loss_fn,
        )

        # Baseline evaluation at t=0
        base_res = self._format_result(
            unique_smis=unique_smis,
            b_vec=b_vec,
            C_mat=C_mat,
            best_alpha=best_alpha,
            ing_meta=ing_meta,
            P_mat=P_mat,
            Delta_tensor=Delta_tensor,
            threshold=threshold,
            target_targets={},
            suppress_indices=suppress_indices,
            mode="perfume",
        )

        # Timeline trajectory evaluation for all keyframe hours
        all_labels = self.vocab.labels
        timeline_evaluation = []

        with torch.no_grad():
            w_0_opt = b_vec + torch.matmul(best_alpha, C_mat.t())
            w_0_opt = torch.clamp(w_0_opt, 0.0, 1.0)
            if torch.sum(w_0_opt) > 0:
                w_0_opt = w_0_opt / torch.sum(w_0_opt)

            for t_float, t_dict in parsed_keyframes:
                decay = 2.0 ** (-t_float / half_lives)
                m_t = w_0_opt * decay
                rem_mass = float(torch.sum(m_t).item())
                w_t = m_t / (rem_mass + 1e-8)

                preds_t = self._predict_from_molecule_weights(w_t.unsqueeze(0), P_mat, Delta_tensor)[0]
                prob_list = preds_t.tolist()

                active_desc = [
                    {"descriptor": all_labels[i], "probability": round(prob_list[i], 4)}
                    for i in range(len(all_labels))
                    if prob_list[i] >= threshold
                ]
                active_desc.sort(key=lambda x: x["probability"], reverse=True)

                target_achieved = {}
                for idx_d, target_val in t_dict.items():
                    target_achieved[all_labels[idx_d]] = {
                        "target_probability": target_val,
                        "achieved_probability": round(prob_list[idx_d], 4),
                    }

                timeline_evaluation.append({
                    "time_hours": t_float,
                    "remaining_mass_percentage": round(rem_mass * 100.0, 2),
                    "target_notes_achieved": target_achieved,
                    "predicted_descriptors": [d["descriptor"] for d in active_desc],
                    "predictions": active_desc[:8],
                })

        base_res["timeline_evaluation"] = timeline_evaluation
        return base_res

    def _solve_optimal_alpha_direct_weights(
        self,
        K: int,
        W_rem: float,
        loss_fn,
    ) -> torch.Tensor:
        """
        Solves optimal alpha when loss_fn takes alpha directly: loss_fn(alphas: [Batch, K]) -> [Batch].
        Used for multi-time evaluations (timeline targets and temporal quadrature).
        """
        if K == 1:
            return torch.tensor([W_rem], dtype=torch.float32, device=self.device)

        if K == 2:
            t = torch.linspace(0.0, 1.0, 1001, dtype=torch.float32, device=self.device)
            alphas = torch.stack([t * W_rem, (1.0 - t) * W_rem], dim=1)  # [1001, 2]
            losses = loss_fn(alphas)
            best_idx = torch.argmin(losses)
            return alphas[best_idx].detach()

        NUM_RESTARTS = 64
        NUM_STEPS = 80
        logits = torch.randn(NUM_RESTARTS, K, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.15)

        for step in range(NUM_STEPS):
            optimizer.zero_grad()
            alphas = F.softmax(logits, dim=-1) * W_rem
            losses = loss_fn(alphas)
            loss_total = torch.sum(losses)
            loss_total.backward()
            optimizer.step()

        with torch.no_grad():
            alphas_final = F.softmax(logits, dim=-1) * W_rem
            final_losses = loss_fn(alphas_final)
            best_restart_idx = torch.argmin(final_losses)
            return alphas_final[best_restart_idx].detach()

    def _solve_optimal_alpha(
        self,
        K: int,
        W_rem: float,
        b_vec: torch.Tensor,
        C_mat: torch.Tensor,
        P_mat: torch.Tensor,
        Delta_tensor: torch.Tensor,
        loss_fn,
    ) -> torch.Tensor:
        """Finds alpha in [0, W_rem]^K summing to W_rem that minimizes loss_fn."""
        if K == 1:
            return torch.tensor([W_rem], dtype=torch.float32, device=self.device)

        if K == 2:
            # Exact 1D vectorized line search with 1,001 points (0.1% resolution)
            t = torch.linspace(0.0, 1.0, 1001, dtype=torch.float32, device=self.device)  # [1001]
            alphas = torch.stack([t * W_rem, (1.0 - t) * W_rem], dim=1)  # [1001, 2]
            # w = b + C @ alpha -> [1001, M]
            w_batch = b_vec.unsqueeze(0) + torch.matmul(alphas, C_mat.t())  # [1001, M]
            preds = self._predict_from_molecule_weights(w_batch, P_mat, Delta_tensor)  # [1001, D]
            losses = loss_fn(preds)
            best_idx = torch.argmin(losses)
            return alphas[best_idx].detach()

        # For K >= 3: Multi-start Simplex Adam with 64 parallel restarts
        NUM_RESTARTS = 64
        NUM_STEPS = 80
        # Initialize logits uniformly and randomly on simplex
        logits = torch.randn(NUM_RESTARTS, K, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.15)

        for step in range(NUM_STEPS):
            optimizer.zero_grad()
            alphas = F.softmax(logits, dim=-1) * W_rem  # [NUM_RESTARTS, K]
            w_batch = b_vec.unsqueeze(0) + torch.matmul(alphas, C_mat.t())  # [NUM_RESTARTS, M]
            preds = self._predict_from_molecule_weights(w_batch, P_mat, Delta_tensor)
            losses = loss_fn(preds)
            loss_total = torch.sum(losses)
            loss_total.backward()
            optimizer.step()

        # Evaluate final loss across all restarts
        with torch.no_grad():
            alphas_final = F.softmax(logits, dim=-1) * W_rem
            w_final = b_vec.unsqueeze(0) + torch.matmul(alphas_final, C_mat.t())
            preds_final = self._predict_from_molecule_weights(w_final, P_mat, Delta_tensor)
            final_losses = loss_fn(preds_final)
            best_restart_idx = torch.argmin(final_losses)
            return alphas_final[best_restart_idx].detach()

    def _format_result(
        self,
        unique_smis: List[str],
        b_vec: torch.Tensor,
        C_mat: torch.Tensor,
        best_alpha: torch.Tensor,
        ing_meta: List[Dict[str, Any]],
        P_mat: torch.Tensor,
        Delta_tensor: torch.Tensor,
        threshold: float,
        target_targets: Dict[int, float],
        suppress_indices: List[int],
        mode: str,
    ) -> Dict[str, Any]:
        """Constructs detailed human and machine readable output dict."""
        with torch.no_grad():
            w_opt = b_vec + torch.matmul(best_alpha, C_mat.t())
            w_opt = torch.clamp(w_opt, 0.0, 1.0)
            # Re-normalize slightly if float imprecision
            sum_w = torch.sum(w_opt).item()
            if sum_w > 0:
                w_opt = w_opt / sum_w

            pred_probs = self._predict_from_molecule_weights(w_opt.unsqueeze(0), P_mat, Delta_tensor)[0]
            prob_list = pred_probs.tolist()

        # Build recipe dictionary
        mol_recipe = {}
        for smi, w in zip(unique_smis, w_opt.tolist()):
            if w > 1e-4:
                mol_recipe[smi] = w

        final_ratio_str = format_ratio_expression_from_dict(mol_recipe, round_digits=4)

        # Ingredient percentage breakdown
        ingredient_breakdown = []
        for idx, ing in enumerate(ing_meta):
            weight_val = float(best_alpha[idx].item())
            ingredient_breakdown.append({
                "ingredient": ing["raw"],
                "percentage": round(weight_val * 100.0, 2),
                "fraction": round(weight_val, 4),
            })

        base_percentage = round(float(torch.sum(b_vec).item()) * 100.0, 2)

        # Active descriptors
        all_labels = self.vocab.labels
        active_descriptors = [
            {"descriptor": all_labels[i], "probability": round(prob_list[i], 4)}
            for i in range(len(all_labels))
            if prob_list[i] >= threshold
        ]
        active_descriptors.sort(key=lambda x: x["probability"], reverse=True)

        res: Dict[str, Any] = {
            "mode": mode,
            "optimal_mixture_string": final_ratio_str,
            "base_percentage": base_percentage,
            "ingredient_percentages": ingredient_breakdown,
            "predicted_descriptors": [p["descriptor"] for p in active_descriptors],
            "predictions": active_descriptors,
        }

        if mode == "balance":
            targets_summary = {}
            for idx_d, target_val in target_targets.items():
                desc_name = all_labels[idx_d]
                targets_summary[desc_name] = {
                    "target_probability": target_val,
                    "achieved_probability": round(prob_list[idx_d], 4),
                }
            suppress_summary = {}
            for idx_s in suppress_indices:
                desc_name = all_labels[idx_s]
                suppress_summary[desc_name] = {
                    "achieved_probability": round(prob_list[idx_s], 4),
                }
            res["target_notes_achieved"] = targets_summary
            res["suppressed_notes_achieved"] = suppress_summary

        elif mode == "neutralize":
            odorless_idx = all_labels.index("odorless") if "odorless" in all_labels else -1
            odorless_prob = round(prob_list[odorless_idx], 4) if odorless_idx >= 0 else 0.0
            other_active = [p for p in active_descriptors if p["descriptor"] != "odorless"]

            res["odorless_probability"] = odorless_prob
            res["is_fully_neutralized"] = (len(other_active) == 0) or (odorless_prob >= 0.5 and len(other_active) <= 1)
            res["residual_scents"] = other_active

        return res


_OPTIMIZER_INSTANCE: Optional[MixtureOptimizer] = None


def get_optimizer() -> MixtureOptimizer:
    global _OPTIMIZER_INSTANCE
    if _OPTIMIZER_INSTANCE is None:
        _OPTIMIZER_INSTANCE = MixtureOptimizer()
    return _OPTIMIZER_INSTANCE
