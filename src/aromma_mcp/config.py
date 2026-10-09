from pathlib import Path
import os
import shutil
import logging
import requests
from tqdm import tqdm

logger = logging.getLogger("aromma_mcp.config")

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AROMMA_SUBMODULE_DIR = PROJECT_ROOT / "aromma"
AROMMA_MODELS_DIR = AROMMA_SUBMODULE_DIR / "models"
AROMMA_SPMM_DIR = AROMMA_MODELS_DIR / "spmm"

# Checkpoint configurations
CHECKPOINT_PATH = PROJECT_ROOT / "aromma_best_fold.pt"
HUGGINGFACE_CHECKPOINT_URL = (
    "https://huggingface.co/riverallzero/aromma/resolve/main/aromma_best_fold.pt"
)

# Reference fallback path if excluded-references exists locally
LOCAL_REFERENCE_CHECKPOINT = (
    PROJECT_ROOT / "excluded-references" / "aromma-agent" / "aromma" / "aromma_best_fold.pt"
)

# SPMM Vocabulary and Bert Config files
SPMM_VOCAB_PATH = AROMMA_SPMM_DIR / "vocab_bpe_300.txt"
SPMM_CONFIG_PATH = AROMMA_SPMM_DIR / "config_bert.json"

# Vocabulary labels (152 descriptors)
LABELS_PATH = AROMMA_SUBMODULE_DIR / "data" / "labels_152.txt"

# LanceDB vector database directory
DATA_DIR = PROJECT_ROOT / "data"
LANCEDB_DIR = DATA_DIR / "lancedb"

# Raw mixture dataset fallback
REFERENCE_MIXTURE_CSV = (
    PROJECT_ROOT / "excluded-references" / "aromma-agent" / "data" / "mixture.csv"
)


def ensure_spmm_files():
    """Ensures SPMM vocab and config are present in aromma/models/spmm."""
    AROMMA_SPMM_DIR.mkdir(parents=True, exist_ok=True)
    ref_spmm_dir = PROJECT_ROOT / "excluded-references" / "aromma-agent" / "aromma" / "spmm"
    if not SPMM_VOCAB_PATH.exists() and (ref_spmm_dir / "vocab_bpe_300.txt").exists():
        shutil.copyfile(ref_spmm_dir / "vocab_bpe_300.txt", SPMM_VOCAB_PATH)
    if not SPMM_CONFIG_PATH.exists() and (ref_spmm_dir / "config_bert.json").exists():
        shutil.copyfile(ref_spmm_dir / "config_bert.json", SPMM_CONFIG_PATH)


def ensure_checkpoint() -> Path:
    """
    Ensures that aromma_best_fold.pt exists in the project root.
    If missing, it first attempts to copy from local excluded-references,
    and if not available, downloads it directly from Hugging Face.
    """
    ensure_spmm_files()

    if CHECKPOINT_PATH.exists() and CHECKPOINT_PATH.stat().st_size > 0:
        return CHECKPOINT_PATH

    # Check local excluded-references fallback
    if LOCAL_REFERENCE_CHECKPOINT.exists() and LOCAL_REFERENCE_CHECKPOINT.stat().st_size > 0:
        logger.info(f"Copying checkpoint from {LOCAL_REFERENCE_CHECKPOINT} to {CHECKPOINT_PATH}...")
        shutil.copyfile(LOCAL_REFERENCE_CHECKPOINT, CHECKPOINT_PATH)
        return CHECKPOINT_PATH

    # Download from Hugging Face
    logger.info(f"Downloading checkpoint from {HUGGINGFACE_CHECKPOINT_URL} to {CHECKPOINT_PATH}...")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CHECKPOINT_PATH.with_suffix(".tmp")
    
    response = requests.get(HUGGINGFACE_CHECKPOINT_URL, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(temp_path, "wb") as f, tqdm(
        desc="Downloading aromma_best_fold.pt",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    os.replace(temp_path, CHECKPOINT_PATH)
    logger.info("Checkpoint download complete.")
    return CHECKPOINT_PATH
