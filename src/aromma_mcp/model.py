import sys
import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from rdkit import Chem
from transformers import BertTokenizer, WordpieceTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from aromma_mcp.config import (
    AROMMA_SUBMODULE_DIR,
    CHECKPOINT_PATH,
    SPMM_VOCAB_PATH,
    SPMM_CONFIG_PATH,
    ensure_checkpoint,
)
from aromma_mcp.vocab import get_vocabulary
from aromma_mcp.utils.chemistry import (
    canonicalize_single_smiles,
    parse_ratio_expression,
    canonical_ratio_string,
    canonical_pair_key,
)
from aromma_mcp.db import get_db

logger = logging.getLogger("aromma_mcp.model")

submodule_str = str(AROMMA_SUBMODULE_DIR)
if submodule_str not in sys.path:
    sys.path.insert(0, submodule_str)

import importlib.util

def _load_submodule_file(module_name: str, rel_path: str):
    file_path = AROMMA_SUBMODULE_DIR / rel_path
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_xbert = _load_submodule_file("aromma_xbert", "src/model/spmm/xbert.py")
BertConfig = _xbert.BertConfig
BertForMaskedLM = _xbert.BertForMaskedLM

_framework = _load_submodule_file("aromma_framework", "src/model/framework.py")
EndToEndModel = _framework.EndToEndModel


class SPMMInference(nn.Module):
    """Submodule-aligned SPMM molecule embedder for inference."""

    def __init__(self, r: int = 4, lora_alpha: int = 8, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not SPMM_VOCAB_PATH.exists():
            raise FileNotFoundError(f"SPMM vocab not found at {SPMM_VOCAB_PATH}")
        if not SPMM_CONFIG_PATH.exists():
            raise FileNotFoundError(f"SPMM config not found at {SPMM_CONFIG_PATH}")

        self.tokenizer = BertTokenizer(
            vocab_file=str(SPMM_VOCAB_PATH),
            do_lower_case=False,
            do_basic_tokenize=False,
        )
        self.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.tokenizer.vocab,
            unk_token=self.tokenizer.unk_token,
            max_input_chars_per_word=250,
        )

        bert_config = BertConfig.from_json_file(str(SPMM_CONFIG_PATH))
        self.text_encoder = BertForMaskedLM(config=bert_config)
        for i in range(bert_config.fusion_layer, bert_config.num_hidden_layers):
            self.text_encoder.bert.encoder.layer[i] = nn.Identity()
        self.text_encoder.cls = nn.Identity()

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            task_type=TaskType.SEQ_CLS,
        )
        self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        self.to(self.device)
        self.output_dim = 768

    def tokenize(self, smiles_list: List[str]):
        tokens = [
            "[CLS]"
            + Chem.MolToSmiles(
                Chem.MolFromSmiles(s),
                isomericSmiles=False,
                canonical=True,
            )
            for s in smiles_list
        ]
        tok = self.tokenizer(
            tokens,
            padding="longest",
            truncation=True,
            max_length=100,
            return_tensors="pt",
        ).to(self.device)
        return tok

    def forward(self, smiles_list: List[str]):
        tok = self.tokenize(smiles_list)
        logits = self.text_encoder.bert(
            tok.input_ids[:, 1:],
            attention_mask=tok.attention_mask[:, 1:],
            return_dict=True,
            mode="text",
        ).last_hidden_state[:, 0, :]
        return logits


class ArommaPredictor:
    """
    High-level neural odor predictor with:
    - Pure constituent caching in 'molecules'
    - Pairwise synergy delta caching in 'pairwise_interactions'
    - Instant ratio composition via Neural Synergy-Aware Mixture Interpolation
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = get_vocabulary()
        self.db = get_db()
        self._load_model()

    def _load_model(self):
        ckpt_path = ensure_checkpoint()
        logger.info(f"Loading AROMMA model on {self.device} using weights from {ckpt_path}...")

        self.embedder = SPMMInference(r=4, lora_alpha=8, device=self.device)
        self.model = EndToEndModel(
            embedder=self.embedder,
            sattn_hidden_dim=196,
            cattn_hidden_dim=384,
            num_heads=4,
            num_labels=len(self.vocab),
        ).to(self.device)

        state_dict = torch.load(str(ckpt_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("AROMMA model successfully loaded.")

    @torch.no_grad()
    def get_or_predict_single(self, smiles: str) -> List[float]:
        """
        Retrieves probabilities for a single molecule from 'molecules' table.
        If missing, executes forward pass on GPU and writes to table.
        """
        csmi = canonicalize_single_smiles(smiles)
        if not csmi:
            raise ValueError(f"Invalid SMILES string: '{smiles}'")

        cached = self.db.get_molecule(csmi)
        if cached is not None:
            return [float(p) for p in cached["probabilities"]]

        # Forward pass
        logits = self.model.forward([csmi])
        prob_vec = torch.sigmoid(logits[0]).cpu().tolist()

        active = [
            self.vocab.labels[i]
            for i, p in enumerate(prob_vec)
            if p >= 0.3
        ]
        self.db.upsert_molecule(csmi, active, prob_vec, source="prediction")
        return prob_vec

    @torch.no_grad()
    def get_or_predict_interaction(self, smi_a: str, smi_b: str) -> List[float]:
        """
        Retrieves pairwise synergy delta Delta_ab from 'pairwise_interactions' table.
        If missing, evaluates 1:1 mixture on GPU, computes delta, and writes to table.
        """
        ca, cb, pair_key = canonical_pair_key(smi_a, smi_b)
        cached = self.db.get_interaction(ca, cb)
        if cached is not None:
            return [float(d) for d in cached["delta_synergy"]]

        p_a = torch.tensor(self.get_or_predict_single(ca), device=self.device)
        p_b = torch.tensor(self.get_or_predict_single(cb), device=self.device)

        # 1:1 mixture forward pass on GPU
        mix_str = f"{ca};{cb}"
        logits = self.model.forward([mix_str])
        p_mix = torch.sigmoid(logits[0])

        delta = (p_mix - 0.5 * (p_a + p_b)).cpu().tolist()
        self.db.upsert_interaction(ca, cb, delta)
        return delta

    def predict(
        self,
        smiles: str,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Predicts odor descriptors for any molecule or mixture with optional ratios.
        E.g. 'CCN', 'CCN:0.7;CCO:0.3', 'CCN:1;CCO:2;CC(=O)O:1'
        """
        parsed = parse_ratio_expression(smiles)
        if not parsed:
            raise ValueError(f"Invalid SMILES or ratio expression: '{smiles}'")

        smiles_list, weights = parsed
        all_labels = self.vocab.labels
        N = len(smiles_list)

        # 1. Single Molecule
        if N == 1:
            csmi = smiles_list[0]
            prob_vec = self.get_or_predict_single(csmi)
            canon_str = csmi
        else:
            # 2. Multi-Component Mixture via Neural Synergy Interpolation
            p_components = [
                torch.tensor(self.get_or_predict_single(smi))
                for smi in smiles_list
            ]

            # Linear sum: sum(w_i * P_i)
            blended = sum(w * p for w, p in zip(weights, p_components))

            # Pairwise non-linear synergy: sum_{i < j} 4 * w_i * w_j * Delta_ij
            for i in range(N):
                for j in range(i + 1, N):
                    w_ij = weights[i] * weights[j]
                    if w_ij > 0:
                        delta = torch.tensor(self.get_or_predict_interaction(smiles_list[i], smiles_list[j]))
                        blended += 4.0 * w_ij * delta

            blended = torch.clamp(blended, 0.0, 1.0)
            prob_vec = blended.tolist()
            canon_str = canonical_ratio_string(smiles)

        predictions = [
            {"descriptor": label, "probability": float(prob_vec[i])}
            for i, label in enumerate(all_labels)
        ]
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        active_descriptors = [p for p in predictions if p["probability"] >= threshold]

        top_preds = predictions[:top_k] if (top_k and top_k > 0) else active_descriptors

        return {
            "input_smiles": smiles,
            "canonical_smiles": canon_str,
            "components": smiles_list,
            "weights": weights,
            "threshold": threshold,
            "predicted_descriptors": [p["descriptor"] for p in active_descriptors],
            "predictions": top_preds,
        }

    @torch.no_grad()
    def predict_mixture_batch(self, mixtures: List[str]) -> List[List[float]]:
        """Batch GPU inference for list of raw mixture strings."""
        if not mixtures:
            return []
        logits = self.model.forward(mixtures)
        return torch.sigmoid(logits).cpu().tolist()

    @torch.no_grad()
    def ensure_molecules_and_interactions_cached(self, unique_smiles: List[str]):
        """
        Ensures all constituent molecules and pairwise interactions are present in LanceDB.
        Performs GPU forward passes in batches (chunks of 256) and commits missing pairs
        in a single batch commit.
        """
        canon_mols = []
        for s in unique_smiles:
            c = canonicalize_single_smiles(s)
            if c and c not in canon_mols:
                canon_mols.append(c)

        # 1. Batch fetch & predict missing single molecules
        missing_mols = []
        for smi in canon_mols:
            if self.db.get_molecule(smi) is None:
                missing_mols.append(smi)

        if missing_mols:
            logger.info(f"Computing predictions for {len(missing_mols)} missing single molecules in batches...")
            batch_size = 256
            for i in range(0, len(missing_mols), batch_size):
                chunk = missing_mols[i:i + batch_size]
                logits = self.model.forward(chunk)
                probs = torch.sigmoid(logits).cpu().tolist()
                for smi, p_vec in zip(chunk, probs):
                    active = [self.vocab.labels[k] for k, p in enumerate(p_vec) if p >= 0.3]
                    self.db.upsert_molecule(smi, active, p_vec, source="prediction")

        # 2. Check pairwise interactions
        M = len(canon_mols)
        missing_pairs = []
        for i in range(M):
            for j in range(i + 1, M):
                ca, cb, _ = canonical_pair_key(canon_mols[i], canon_mols[j])
                if self.db.get_interaction(ca, cb) is None:
                    missing_pairs.append((ca, cb))

        if missing_pairs:
            logger.info(f"Computing {len(missing_pairs)} missing pairwise interactions in batches of 256...")
            batch_size = 256
            items_to_save = []
            for i in range(0, len(missing_pairs), batch_size):
                chunk = missing_pairs[i:i + batch_size]
                mix_strings = [f"{ca};{cb}" for ca, cb in chunk]
                logits = self.model.forward(mix_strings)
                probs_mix = torch.sigmoid(logits).cpu()

                for (ca, cb), p_mix in zip(chunk, probs_mix):
                    p_a = torch.tensor(self.get_or_predict_single(ca))
                    p_b = torch.tensor(self.get_or_predict_single(cb))
                    delta = (p_mix - 0.5 * (p_a + p_b)).tolist()
                    items_to_save.append({
                        "smiles_a": ca,
                        "smiles_b": cb,
                        "delta_synergy": delta,
                    })

            self.db.upsert_interactions_batch(items_to_save)
            logger.info(f"Cached {len(items_to_save)} pairwise interactions.")


_PREDICTOR_INSTANCE: Optional[ArommaPredictor] = None


def get_predictor() -> ArommaPredictor:
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = ArommaPredictor()
    return _PREDICTOR_INSTANCE
