import pandas as pd
from typing import List, Tuple


def load_sentiment_csv(path: str) -> Tuple[List[str], List[int]]:
    """
    Assumes:
      - CSV with exactly two useful columns:
        * first  column = label (string, e.g. 'negative')
        * second column = text  (string)
      - exactly 3 distinct labels overall.
    """
    df = pd.read_csv(path)

    # columns are not properly labeled 
    label_col, text_col = df.columns[:2]

    texts = df[text_col].astype(str).tolist()
    raw_labels = df[label_col].astype(str).tolist()

    label2id = {}
    labels: List[int] = []

    for lab in raw_labels:
        if lab not in label2id:
            label2id[lab] = len(label2id)
        labels.append(label2id[lab])

    if len(label2id) != 3:
        raise ValueError(f"Expected 3 labels, found {len(label2id)}: {label2id}")

    return texts, labels
