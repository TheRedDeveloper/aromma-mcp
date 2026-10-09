from typing import List, Optional
from aromma_mcp.config import LABELS_PATH


class VocabularyManager:
    """Manages the 152 standard AROMMA olfactory descriptors."""

    def __init__(self, labels_file=LABELS_PATH):
        self.labels_file = labels_file
        self._labels: List[str] = []
        self._load_labels()

    def _load_labels(self):
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found at: {self.labels_file}")
        with open(self.labels_file, "r", encoding="utf-8") as f:
            self._labels = [line.strip().lower() for line in f if line.strip()]

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    def contains(self, descriptor: str) -> bool:
        return descriptor.strip().lower() in self._labels

    def search(self, query: str) -> List[str]:
        """Search descriptors matching a substring."""
        query = query.strip().lower()
        return [lbl for lbl in self._labels if query in lbl]


_VOCAB_INSTANCE: Optional[VocabularyManager] = None


def get_vocabulary() -> VocabularyManager:
    global _VOCAB_INSTANCE
    if _VOCAB_INSTANCE is None:
        _VOCAB_INSTANCE = VocabularyManager()
    return _VOCAB_INSTANCE
