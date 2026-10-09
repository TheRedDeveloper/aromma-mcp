import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import lancedb
import pyarrow as pa
import pandas as pd

from aromma_mcp.config import (
    LANCEDB_DIR,
    REFERENCE_MIXTURE_CSV,
    AROMMA_SUBMODULE_DIR,
)
from aromma_mcp.vocab import get_vocabulary
from aromma_mcp.utils.chemistry import canonicalize_single_smiles, canonical_pair_key

logger = logging.getLogger("aromma_mcp.db")

MOL_TABLE_NAME = "molecules"
PAIR_TABLE_NAME = "pairwise_interactions"

# PyArrow Schema for 'molecules' table (single molecules only)
MOL_SCHEMA = pa.schema([
    pa.field("smiles", pa.string()),
    pa.field("descriptors", pa.list_(pa.string())),
    pa.field("probabilities", pa.list_(pa.float32())),
    pa.field("descriptor_count", pa.int32()),
    pa.field("source", pa.string()),
])

# PyArrow Schema for 'pairwise_interactions' table (synergy deltas)
PAIR_SCHEMA = pa.schema([
    pa.field("pair_key", pa.string()),  # Format: "smiles_a;smiles_b" where a < b
    pa.field("smiles_a", pa.string()),
    pa.field("smiles_b", pa.string()),
    pa.field("delta_synergy", pa.list_(pa.float32())),
])


class OdorDatabase:
    """
    LanceDB-backed dual-table storage:
    - 'molecules': pure single constituent molecules.
    - 'pairwise_interactions': binary neural synergy vectors.
    """

    def __init__(self, db_path: Path = LANCEDB_DIR):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self.mol_table = self._init_mol_table()
        self.pair_table = self._init_pair_table()

    def _init_mol_table(self):
        try:
            table = self.db.open_table(MOL_TABLE_NAME)
            logger.info(f"Opened existing LanceDB table '{MOL_TABLE_NAME}'")
            return table
        except Exception:
            logger.info(f"Creating new LanceDB table '{MOL_TABLE_NAME}'")
            table = self.db.create_table(MOL_TABLE_NAME, schema=MOL_SCHEMA)
            self._populate_initial_benchmark(table)
            return table

    def _init_pair_table(self):
        try:
            table = self.db.open_table(PAIR_TABLE_NAME)
            logger.info(f"Opened existing LanceDB table '{PAIR_TABLE_NAME}'")
            return table
        except Exception:
            logger.info(f"Creating new LanceDB table '{PAIR_TABLE_NAME}'")
            return self.db.create_table(PAIR_TABLE_NAME, schema=PAIR_SCHEMA)

    def _populate_initial_benchmark(self, table):
        """Populates benchmark single molecules into the molecules table."""
        logger.info("Extracting verified single molecules for LanceDB initialization...")
        molecules: Dict[str, Set[str]] = {}

        if REFERENCE_MIXTURE_CSV.exists():
            df = pd.read_csv(REFERENCE_MIXTURE_CSV)
            for _, row in df.iterrows():
                raw_smi = str(row.get("smiles", ""))
                raw_desc = str(row.get("descriptors", ""))
                if ";" in raw_smi or not raw_smi.strip():
                    continue
                csmi = canonicalize_single_smiles(raw_smi)
                if not csmi:
                    continue
                descriptors = {
                    d.strip().lower()
                    for d in raw_desc.split(";")
                    if d.strip()
                }
                if csmi not in molecules:
                    molecules[csmi] = set()
                molecules[csmi].update(descriptors)

        train_csv = AROMMA_SUBMODULE_DIR / "data" / "mixture" / "fold1" / "train.csv"
        if train_csv.exists():
            df_fold = pd.read_csv(train_csv)
            desc_cols = [c for c in df_fold.columns if c != "smiles"]
            for _, row in df_fold.iterrows():
                raw_smi = str(row.get("smiles", ""))
                if ";" in raw_smi or not raw_smi.strip():
                    continue
                csmi = canonicalize_single_smiles(raw_smi)
                if not csmi:
                    continue
                active_descriptors = {
                    col.strip().lower()
                    for col in desc_cols
                    if row.get(col, 0) == 1
                }
                if csmi not in molecules:
                    molecules[csmi] = set()
                molecules[csmi].update(active_descriptors)

        vocab = get_vocabulary()
        labels = vocab.labels
        rows = []
        for smi, desc_set in molecules.items():
            desc_list = sorted(list(desc_set))
            probs = [1.0 if lbl in desc_set else 0.0 for lbl in labels]
            rows.append({
                "smiles": smi,
                "descriptors": desc_list,
                "probabilities": probs,
                "descriptor_count": len(desc_list),
                "source": "benchmark",
            })

        if rows:
            table.add(rows)
            logger.info(f"Populated {len(rows)} benchmark molecules into '{MOL_TABLE_NAME}'.")

    # --- Single Molecule Operations ---

    def get_molecule(self, smiles: str) -> Optional[Dict[str, Any]]:
        """Look up single molecule by canonical SMILES."""
        csmi = canonicalize_single_smiles(smiles)
        if not csmi:
            return None
        escaped = csmi.replace("'", "''")
        res = self.mol_table.search().where(f"smiles = '{escaped}'").limit(1).to_list()
        return res[0] if res else None

    def upsert_molecule(
        self,
        smiles: str,
        descriptors: List[str],
        probabilities: List[float],
        source: str = "prediction",
    ):
        """Insert or update single molecule in 'molecules' table."""
        csmi = canonicalize_single_smiles(smiles)
        if not csmi:
            return
        escaped = csmi.replace("'", "''")
        try:
            self.mol_table.delete(f"smiles = '{escaped}'")
        except Exception:
            pass

        row = [{
            "smiles": csmi,
            "descriptors": sorted(descriptors),
            "probabilities": [float(p) for p in probabilities],
            "descriptor_count": len(descriptors),
            "source": source,
        }]
        self.mol_table.add(row)

    # --- Pairwise Interaction Operations ---

    def get_interaction(self, smiles_a: str, smiles_b: str) -> Optional[Dict[str, Any]]:
        """Look up binary synergy delta by pair key (commutative)."""
        _, _, pair_key = canonical_pair_key(smiles_a, smiles_b)
        escaped = pair_key.replace("'", "''")
        res = self.pair_table.search().where(f"pair_key = '{escaped}'").limit(1).to_list()
        return res[0] if res else None

    def get_interactions_for_molecule(self, target_smiles: str) -> Dict[str, List[float]]:
        """
        Retrieves all cached interactions involving target_smiles in a single query.
        Returns mapping: other_smiles -> delta_synergy
        """
        escaped = target_smiles.replace("'", "''")
        # Matches either smiles_a or smiles_b
        filter_expr = f"smiles_a = '{escaped}' OR smiles_b = '{escaped}'"
        try:
            records = self.pair_table.search().where(filter_expr).limit(50000).to_list()
        except Exception:
            return {}
        mapping = {}
        for r in records:
            other = r["smiles_b"] if r["smiles_a"] == target_smiles else r["smiles_a"]
            mapping[other] = r["delta_synergy"]
        return mapping

    def upsert_interaction(
        self,
        smiles_a: str,
        smiles_b: str,
        delta_synergy: List[float],
    ):
        """Insert or update binary synergy delta in 'pairwise_interactions' table."""
        smi_min, smi_max, pair_key = canonical_pair_key(smiles_a, smiles_b)
        escaped = pair_key.replace("'", "''")
        try:
            self.pair_table.delete(f"pair_key = '{escaped}'")
        except Exception:
            pass

        row = [{
            "pair_key": pair_key,
            "smiles_a": smi_min,
            "smiles_b": smi_max,
            "delta_synergy": [float(d) for d in delta_synergy],
        }]
        self.pair_table.add(row)

    def upsert_interactions_batch(self, items: List[Dict[str, Any]]):
        """
        Batch-inserts pairwise synergy deltas in a single LanceDB commit.
        items: list of dicts with keys: 'smiles_a', 'smiles_b', 'delta_synergy'
        """
        if not items:
            return
        rows = []
        for it in items:
            smi_min, smi_max, pair_key = canonical_pair_key(it["smiles_a"], it["smiles_b"])
            rows.append({
                "pair_key": pair_key,
                "smiles_a": smi_min,
                "smiles_b": smi_max,
                "delta_synergy": [float(d) for d in it["delta_synergy"]],
            })
        self.pair_table.add(rows)

    # --- Search Retrieval ---

    def get_searchable_molecules(
        self,
        additional_smiles: Optional[List[str]] = None,
        include_database: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves candidate molecules according to include_database and additional_smiles flags.
        """
        results = []
        if include_database:
            results.extend(self.mol_table.search().limit(50000).to_list())

        if additional_smiles:
            from aromma_mcp.utils.chemistry import parse_ratio_expression
            for item in additional_smiles:
                parsed = parse_ratio_expression(item)
                if not parsed:
                    continue
                csmi = parsed[0][0] if len(parsed[0]) == 1 else item
                existing = next((r for r in results if r["smiles"] == csmi), None)
                if not existing:
                    mol_entry = self.get_molecule(csmi)
                    if mol_entry:
                        results.append(mol_entry)

        return results

    # Compatibility methods
    def get(self, smiles: str) -> Optional[Dict[str, Any]]:
        return self.get_molecule(smiles)

    def upsert(self, smiles: str, descriptors: List[str], probabilities: List[float], source: str = "prediction"):
        self.upsert_molecule(smiles, descriptors, probabilities, source)

    def count(self) -> int:
        return self.mol_table.count_rows()


_DB_INSTANCE: Optional[OdorDatabase] = None


def get_db() -> OdorDatabase:
    global _DB_INSTANCE
    if _DB_INSTANCE is None:
        _DB_INSTANCE = OdorDatabase()
    return _DB_INSTANCE
