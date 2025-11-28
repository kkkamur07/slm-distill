"""Simple dataset class for parquet data"""

import os
import pandas as pd
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset


class NativeSLMData(TorchDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int,
        train: bool = True,
        train_split: float = 0.95,
        seed: int = 42,
        cache_dir: str = None
    ):
        cache_path = None

        split = "train" if train else "val"
        
        if cache_dir:
            cache_path = os.path.join(
                cache_dir, 
                f"{os.path.basename(data_path)}_{split}_ml{max_length}"
            )
            
            if os.path.exists(cache_path):
                self.dataset = load_from_disk(cache_path).with_format("torch")
                return
        
        # Need to load it in chunks to avoid memory issues for the files.
        df = pd.read_parquet(data_path)
        dataset = Dataset.from_pandas(df) # Problem is here.
        
        split_dataset = dataset.train_test_split(train_size=train_split, seed=seed)
        dataset = split_dataset['train'] if train else split_dataset['test']
        
        self.dataset = dataset.map(
            
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_special_tokens_mask=True,
            ),
            batched=True,
            batch_size=1024,
            remove_columns=dataset.column_names,
            
        ).with_format("torch")
        
        # Save to cache
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Saving to cache: {cache_path}")
            self.dataset.save_to_disk(cache_path)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]