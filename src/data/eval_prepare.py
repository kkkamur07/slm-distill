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