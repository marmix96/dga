import string
from pathlib import Path
from typing import Dict
from typing import Final
from typing import List

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
MODEL_DIR: Final[Path] = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA: Final[Dict[int, List[Path]]] = {
    0: [DATA_DIR / "alexa.csv", DATA_DIR / "cisco.csv"],
    1: [DATA_DIR / "mal.csv"],
}

VOWELS: Final[List[str]] = list("aeiouy")
CONSONANTS: Final[List[str]] = [c for c in string.ascii_lowercase if c not in VOWELS]
HYPHENS: Final[List[str]] = ["-"]
DIGITS: Final[List[str]] = [str(i) for i in range(0, 10)]

VOCAB: Final[List[str]] = sorted(VOWELS + CONSONANTS + HYPHENS + DIGITS + [".", "_"])

TRAIN_EXPLAINABILITY_SAMPLES = 1000
BINARY_EXPLAINABILITY_SAMPLES = 1000
MULTICLASS_EXPLAINABILITY_SAMPLES = 200
