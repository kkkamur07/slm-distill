import json
from typing import Dict, List, Tuple


def load_nli_split(
    path: str,
    split_key: str | None = None,
    raw_to_id: Dict[int, int] | None = None,
) -> Tuple[List[str], List[str], List[int], Dict[int, int]]:
    """
    Load an NLI split from JSON.

    Expected example format inside the chosen split/list:
        {"premise": "...", "hypothesis": "...", "label": 2}

    Supported top-level JSON formats:
      * {"train": [...]} / {"dev": [...]} / {"validation": [...]} / {"test": [...]} / {"data": [...]}
      * {"some_single_key": [...]}            # if split_key is None and dict has one key
      * [ {...}, {...}, ... ]                # plain list

    raw_to_id:
      * if None, we build a mapping sorted(set(raw_labels)) -> [0..K-1]
      * if provided, we reuse it so that all splits share the same label ids.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = None

    if isinstance(data, dict):
        # If explicit split_key is given and present, use that
        if split_key is not None and split_key in data:
            records = data[split_key]
        else:
            # Try common split keys
            for k in ("train", "dev", "validation", "test", "data"):
                if k in data:
                    records = data[k]
                    break

            # Fallback: single-key dict
            if records is None:
                if len(data) == 1:
                    records = next(iter(data.values()))
                else:
                    raise ValueError(
                        f"Unexpected JSON structure in {path}. "
                        "Provide split_key or ensure a single top-level key."
                    )
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unexpected JSON structure in {path}: {type(data)}")

    premises: List[str] = []
    hypotheses: List[str] = []
    raw_labels: List[int] = []

    for ex in records:
        if "premise" not in ex or "hypothesis" not in ex or "label" not in ex:
            raise ValueError(f"Missing keys in example: {ex}")
        premises.append(str(ex["premise"]))
        hypotheses.append(str(ex["hypothesis"]))
        raw_labels.append(int(ex["label"]))

    if raw_to_id is None:
        unique_raw = sorted(set(raw_labels))
        raw_to_id = {raw: idx for idx, raw in enumerate(unique_raw)}

    labels = [raw_to_id[r] for r in raw_labels]
    return premises, hypotheses, labels, raw_to_id