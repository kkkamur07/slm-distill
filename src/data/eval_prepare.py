"""Helper functions for preparing evaluation datasets.

This module provides utilities to load and prepare parquet data for evaluation
with masked language modeling. It includes:
- ParquetTextDataset: Simple dataset class for loading parquet files
- prepare_datasets: Convenience function to create DataLoader with MLM collation

Used primarily for quick evaluation and testing during development.

Example:
    >>> from transformers import AutoTokenizer
    >>> from src.data.eval_prepare import prepare_datasets
    >>> 
    >>> tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    >>> eval_loader = prepare_datasets(
    ...     tokenizer=tokenizer,
    ...     data_path="eval_data.parquet",
    ...     max_length=128,
    ...     batch_size=32
    ... )
    >>> for batch in eval_loader:
    ...     # Process evaluation batch
    ...     pass
"""

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling


class ParquetTextDataset(Dataset):
    def __init__(self, tokenizer, path, max_length):
        texts = pd.read_parquet(path)["text"].tolist()
        self.enc = tokenizer(texts, truncation=True, max_length=max_length)

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def prepare_datasets(tokenizer, data_path, max_length, batch_size):
    ds = ParquetTextDataset(tokenizer, data_path, max_length)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)