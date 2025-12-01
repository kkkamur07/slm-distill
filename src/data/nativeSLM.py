"""Simple dataset class for parquet data"""

import os
import pandas as pd
from datasets import Dataset, load_from_disk, concatenate_datasets
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
import gc


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
        
        parquet_files = sorted(Path(data_path).glob("*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {data_path}")
        
        all_dataset = []
        
        for i, file in enumerate(parquet_files):
            print(f"Loading parquet file {i+1}/{len(parquet_files)}: {file}")
            
            df_chunk = pd.read_parquet(file)
            ds_chunk = Dataset.from_pandas(df_chunk)
            
            ds_chunk = ds_chunk.map(
                lambda x : tokenizer(
                    x["text"],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_special_tokens_mask=True,
                ),
                batched=True,
                batch_size=1024,
                remove_columns=ds_chunk.column_names,
            )
            
            all_dataset.append(ds_chunk)
            
            del df_chunk
            gc.collect()
            
        combined_dataset = concatenate_datasets(all_dataset)
        split_dataset = combined_dataset.train_test_split(train_size=train_split, seed=seed)
        self.dataset = (split_dataset['train'] if train else split_dataset['test']).with_format("torch")
        
        # Save to cache
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Saving to cache: {cache_path}")
            self.dataset.save_to_disk(cache_path)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]