# AROMMA FastMCP Server

A high-performance Model Context Protocol (MCP) server for neural odor prediction, persistent LanceDB caching, and direct scent modulation powered by [AROMMA](https://github.com/DGIST-Distributed-AI-Lab/aromma).

---

## Capabilities

1. **`predict_odor`**:
   - Neural olfactory prediction for single molecules using the AROMMA architecture (SPMM + Cross-Attention).
   - Generates multi-label probabilities across 152 unified olfactory descriptors.
   - **Persistent LanceDB Caching**: Stores 152-dimensional probability vectors on disk in `data/lancedb/`. Molecules queried once are served instantaneously (<1ms) on subsequent calls.
2. **`recommend_molecules`**:
   - Fast search for single molecules matching desired target scents (e.g. `['citrus', 'floral', 'fresh']`).
   - Supports strict exclusion filtering (e.g. avoiding `['fishy', 'sulfur']`).
   - Supports user inventory restrictions via `available_smiles: list[str]`.
3. **`modulate_scent`**:
   - Directly answers questions like *"What do I add to make this mixture less fishy?"*.
   - Simulates binary mixtures directly on GPU using AROMMA (`AROMMA.forward([f"{current_smiles};{cand_smiles}"])`).
   - Evaluates probability drops on unwanted notes and gains on desired notes, returning the top counteracting additives.
   - Supports restricting candidate additives to an inventory cabinet (`available_smiles`).
4. **`list_descriptors`**:
   - Complete access and search across the official 152 AROMMA odor descriptors.

---

## Setup & Installation

### Conda Environment (Python 3.11 with native PyTorch CUDA)
```bash
conda env create -f environment.yml
conda activate aromma_mcp_env
pip install -e .
```

---

## Running the Server

### Stdio / Command Line
```bash
aromma-mcp
```
or:
```bash
python -m aromma_mcp.server
```

### FastMCP Dev Inspector
```bash
fastmcp dev src/aromma_mcp/server.py
```

### MCP Client Configuration (Claude Desktop / Cursor / Antigravity)
```json
{
  "mcpServers": {
    "aromma": {
      "command": "conda",
      "args": [
        "run",
        "-n",
        "aromma_mcp_env",
        "python",
        "-m",
        "aromma_mcp.server"
      ]
    }
  }
}
```

---

## Example Tool Invocations

### `modulate_scent` (e.g. "What to add to make ethylamine less fishy?")
- **Input**:
  ```python
  modulate_scent(
      current_smiles="CCN",
      reduce_descriptors=["fishy"],
      boost_descriptors=["citrus", "sweet"],
      top_k=3,
  )
  ```
- **Output**:
  ```json
  {
    "current_smiles": "CCN",
    "top_additives": [
      {
        "additive_smiles": "CCCCCCCCCC=O",
        "additive_descriptors": ["aldehydic", "citrus", "floral", "fruity", "orange", "sweet"],
        "unwanted_probability_change": {
          "fishy": {"before": 0.989, "after": 0.010}
        },
        "reduction_score": 1.0883
      }
    ]
  }
  ```
