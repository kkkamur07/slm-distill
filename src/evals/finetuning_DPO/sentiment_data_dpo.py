import pandas as pd
from typing import List, Tuple, Dict


def load_sentiment_csv(
    path: str,
    label2id: Dict[str, int] | None = None,
    num_labels: int | None = None,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Load a sentiment CSV and map string labels to integer ids.

    Assumes:
      - CSV with at least two useful columns:
        * first  column = label (string, e.g. "negative")
        * second column = text  (string)
      - labels are consistent with an optional ``label2id`` mapping.
      - if ``num_labels`` is given, the total number of distinct labels
        across all calls is checked against it.
    """
    df = pd.read_csv(path)

    # columns are not properly labeled
    label_col, text_col = df.columns[:2]

    texts = df[text_col].astype(str).tolist()
    raw_labels = df[label_col].astype(str).tolist()

    if label2id is None:
        label2id = {}

    # extend mapping with any new labels from this file
    for lab in raw_labels:
        if lab not in label2id:
            label2id[lab] = len(label2id)

    if num_labels is not None and len(label2id) != num_labels:
        raise ValueError(
            f"Expected {num_labels} labels, found {len(label2id)}: {label2id}"
        )

    labels: List[int] = [label2id[lab] for lab in raw_labels]

    return texts, labels, label2id
