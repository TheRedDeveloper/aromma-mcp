from rdkit import Chem
from typing import Optional, Tuple, List


def canonicalize_single_smiles(smiles: str) -> Optional[str]:
    """Validates and canonicalizes a single molecule SMILES string."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    clean = smiles.strip()
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def parse_ratio_expression(expression: str) -> Optional[Tuple[List[str], List[float]]]:
    """
    Parses a ratio expression like:
      - 'CCO' -> (['CCO'], [1.0])
      - 'CN:2' -> (['CN'], [1.0])
      - 'CCN;CCO' -> (['CCN', 'CCO'], [0.5, 0.5])
      - 'CCN:0.5;CCO' -> (['CCN', 'CCO'], [0.333, 0.667])
      - 'CCN:1;CCO:2;CC(=O)O:1' -> (['CCN', 'CCO', 'CC(=O)O'], [0.25, 0.5, 0.25])

    Omitted ratio defaults to 1.0.
    Weights are normalized to sum to 1.0.
    Returns None if any component is invalid.
    """
    if not isinstance(expression, str) or not expression.strip():
        return None

    raw_parts = [p.strip() for p in expression.split(";") if p.strip()]
    if not raw_parts:
        return None

    smiles_list = []
    weights_list = []

    for part in raw_parts:
        if ":" in part:
            smi_raw, weight_raw = part.rsplit(":", 1)
            smi_raw = smi_raw.strip()
            try:
                w = float(weight_raw.strip())
                if w < 0:
                    return None
            except ValueError:
                # If colon was inside SMILES or not a float, treat whole thing as SMILES with default 1.0
                smi_raw = part
                w = 1.0
        else:
            smi_raw = part
            w = 1.0  # Omitted default is 1.0

        canon_smi = canonicalize_single_smiles(smi_raw)
        if canon_smi is None:
            return None

        smiles_list.append(canon_smi)
        weights_list.append(w)

    total_weight = sum(weights_list)
    if total_weight <= 0:
        # Uniform if all zero
        normalized_weights = [1.0 / len(weights_list)] * len(weights_list)
    else:
        normalized_weights = [round(w / total_weight, 4) for w in weights_list]

    # If it's a single molecule, weight is strictly 1.0
    if len(smiles_list) == 1:
        normalized_weights = [1.0]

    return smiles_list, normalized_weights


def canonical_ratio_string(expression: str) -> Optional[str]:
    """
    Returns the canonical string representation of a ratio expression.
    For single molecules: returns pure canonical SMILES (e.g. 'CCO').
    For mixtures: returns sorted components with normalized weights.
    """
    parsed = parse_ratio_expression(expression)
    if not parsed:
        return None
    smiles_list, weights = parsed
    if len(smiles_list) == 1:
        return smiles_list[0]

    # Sort deterministically by canonical SMILES
    paired = sorted(zip(smiles_list, weights), key=lambda x: x[0])
    return ";".join([f"{smi}:{w}" for smi, w in paired])


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Alias for canonical_ratio_string for backward compatibility."""
    return canonical_ratio_string(smiles)


def is_valid_smiles(smiles: str) -> bool:
    """Checks whether a SMILES string or ratio expression is chemically valid."""
    return parse_ratio_expression(smiles) is not None


def canonical_pair_key(smi_a: str, smi_b: str) -> Tuple[str, str, str]:
    """
    Enforces commutative ordering for pairwise interactions (A+B == B+A).
    Returns (min_smiles, max_smiles, 'min;max').
    """
    ca = canonicalize_single_smiles(smi_a) or smi_a
    cb = canonicalize_single_smiles(smi_b) or smi_b
    if ca < cb:
        return ca, cb, f"{ca};{cb}"
    else:
        return cb, ca, f"{cb};{ca}"


def parse_unnormalized_expression(expression: str) -> Optional[Tuple[List[str], List[float]]]:
    """
    Parses a ratio expression WITHOUT auto-normalizing the sum to 1.0.
    For each component:
      - If ratio is explicitly given ('CCN:0.4'), uses 0.4.
      - If omitted ('CCN'), defaults to 1.0.
    Returns (smiles_list, raw_weights_list), or None if invalid.
    """
    if not isinstance(expression, str) or not expression.strip():
        return None

    raw_parts = [p.strip() for p in expression.split(";") if p.strip()]
    if not raw_parts:
        return None

    smiles_list = []
    weights_list = []

    for part in raw_parts:
        if ":" in part:
            smi_raw, weight_raw = part.rsplit(":", 1)
            smi_raw = smi_raw.strip()
            try:
                w = float(weight_raw.strip())
                if w < 0:
                    return None
            except ValueError:
                smi_raw = part
                w = 1.0
        else:
            smi_raw = part
            w = 1.0

        canon_smi = canonicalize_single_smiles(smi_raw)
        if canon_smi is None:
            return None

        smiles_list.append(canon_smi)
        weights_list.append(w)

    return smiles_list, weights_list


def format_ratio_expression_from_dict(ratio_dict: dict, round_digits: int = 4) -> str:
    """
    Given a dict of {canonical_smiles: weight}, returns a canonical ratio string
    with weights rounded and sorted deterministically. Omits zero weights.
    """
    active = [(smi, round(float(w), round_digits)) for smi, w in ratio_dict.items() if round(float(w), round_digits) > 0]
    if not active:
        return ""
    if len(active) == 1 and abs(active[0][1] - 1.0) < 1e-4:
        return active[0][0]
    active.sort(key=lambda x: x[0])
    return ";".join([f"{smi}:{w}" for smi, w in active])


def estimate_molecule_half_life(smiles: str) -> float:
    """
    Estimates the objective physical evaporation half-life (t_1/2 in hours)
    for a molecule in a standard matrix-buffered reference state
    (5-10% dilution in carrier ethanol/water, thin film at 25°C / 298.15 K).

    Physical model:
      - Estimated boiling point Tb via calibrated group contribution (heavy atoms, rings, TPSA, HBD/HBA).
      - Log-volatility modulated relative to ambient temperature (298 K).
      - Lower bound clamped at 0.15 h (~9 min) due to solvent diffusion and boundary layer limits.
      - Upper bound clamped at 72.0 h (3 days) for persistent fixatives/crystals.
    """
    from rdkit.Chem import Descriptors
    csmi = canonicalize_single_smiles(smiles) or smiles
    mol = Chem.MolFromSmiles(csmi)
    if mol is None:
        return 1.0  # Default fallback 1 hour

    heavy = mol.GetNumHeavyAtoms()
    rings = mol.GetRingInfo().NumRings()
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(Descriptors.NumHDonors(mol))
    hba = int(Descriptors.NumHAcceptors(mol))

    # 1. Calibrated boiling point in Kelvin
    tb_k = 190.0 + (19.0 * heavy) + (25.0 * rings) + (0.8 * tpsa) + (35.0 * hbd) + (8.0 * hba)

    # 2. Matrix-buffered log10 half-life in hours
    log_t_half = (tb_k - 425.0) / 75.0
    t_half = 10.0 ** log_t_half

    # 3. Physical matrix bounds: 0.15 h (9 min) to 72.0 h (3 days)
    t_half_clamped = max(0.15, min(72.0, t_half))
    return round(float(t_half_clamped), 3)


def evolve_composition_at_time(
    components: List[str],
    weights: List[float],
    hours: float,
) -> Tuple[List[str], List[float], float]:
    """
    Computes the evolved liquid composition at elapsed time `hours`.
    m_i(t) = w_i(0) * 2^(-t / t_1/2,i)
    Returns:
      - components: list of canonical SMILES
      - normalized_weights: liquid fraction of each component at hour t
      - remaining_mass_fraction: total remaining liquid mass relative to t=0
    """
    if hours <= 0:
        total_w = sum(weights)
        norm_w = [w / total_w for w in weights] if total_w > 0 else weights
        return components, norm_w, 1.0

    masses = []
    for smi, w0 in zip(components, weights):
        t_half = estimate_molecule_half_life(smi)
        # Fraction remaining: 2^(-t / t_half)
        decay_factor = 2.0 ** (-hours / t_half)
        masses.append(w0 * decay_factor)

    total_mass = sum(masses)
    if total_mass <= 1e-9:
        # All completely evaporated, return uniform tiny residual
        norm_w = [1.0 / len(components)] * len(components)
        return components, norm_w, 0.0

    normalized_weights = [round(m / total_mass, 4) for m in masses]
    return components, normalized_weights, round(total_mass, 4)



