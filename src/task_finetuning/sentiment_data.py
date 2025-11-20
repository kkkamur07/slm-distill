from typing import List, Dict, Tuple
import random
import pandas as pd

def load_sentiment_data(
    data_path: str = "data/hin/sentiment_metadata.tsv",
    text_column: str = "REVIEW",
    label_column: str = "LABEL",
) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
    """
    Load sentiment dataset from a TSV (current data set for sentiment)
    Expected columns:
        - text_column: review text (default: 'REVIEW')
        - label_column: sentiment label (default: 'LABEL')

    Returns:
        texts: list of review strings
        labels: list of integer ids
        label2id: mapping from label string -> id
        id2label: mapping from id -> label string
    """
    df = pd.read_csv(data_path, sep="\t")

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Expected columns '{text_column}' and '{label_column}' in {data_path}, "
            f"but got: {list(df.columns)}"
        )

    texts = df[text_column].astype(str).tolist()
    label_strings = df[label_column].astype(str).tolist()

    unique_labels = sorted(set(label_strings))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    labels = [label2id[lbl] for lbl in label_strings]

    return texts, labels, label2id, id2label


def split_sentiment_data(
    texts: List[str],
    labels: List[int],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    List[str], List[int],  # train
    List[str], List[int],  # dev
    List[str], List[int],  # test
]:
    """
    Split (texts, labels) into train/dev/test according to the given ratios.

    All splits share the same label space (labels are already ints).
    """
    assert len(texts) == len(labels), "texts and labels must have same length"

    n = len(texts)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    n_test = n - n_train - n_dev

    train_idx = indices[:n_train]
    dev_idx = indices[n_train : n_train + n_dev]
    test_idx = indices[n_train + n_dev :]

    def subset(idxs):
        return [texts[i] for i in idxs], [labels[i] for i in idxs]

    train_texts, train_labels = subset(train_idx)
    dev_texts, dev_labels = subset(dev_idx)
    test_texts, test_labels = subset(test_idx)

    return (
        train_texts,
        train_labels,
        dev_texts,
        dev_labels,
        test_texts,
        test_labels,
    )
