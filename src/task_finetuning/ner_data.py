from typing import List, Tuple
import pandas as pd


def load_wikiann_split(path: str) -> Tuple[List[List[str]], List[List[int]]]:
    """
    data is in a parquet  file

    Returns:
      sentences: List[List[str]]
      labels:    List[List[int]]
    """
    df = pd.read_parquet(path)

    sentences = [list(map(str, toks)) for toks in df["tokens"].tolist()]
    labels = [[int(x) for x in tags] for tags in df["ner_tags"].tolist()]

    return sentences, labels
