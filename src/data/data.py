"""Data loading and preprocessing for custom parquet dataset"""

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import pandas as pd


def prepare_datasets(
    data_path: str,
    tokenizer,
    max_length: int,
    batch_size: int,
    train_split: float = 0.95,
    num_workers: int = 2
):
    
    print(f"\nLoading data from parquet...")
    print(f"Data path: {data_path}")
    
    # Load parquet file
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df):,} examples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(
        train_size=train_split,
        seed=42
    )
    train_data = split_dataset['train']
    eval_data = split_dataset['test']
    
    print(f"✓ Train: {len(train_data):,} examples ({train_split*100:.1f}%)")
    print(f"✓ Eval: {len(eval_data):,} examples ({(1-train_split)*100:.1f}%)")
    
    # Tokenization function
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )
    
    # Tokenize datasets -> need a way to cache this if you want to experiment more. 
    print("Tokenizing...")
    train_data = train_data.map(
        tokenize,
        batched=True,
        batch_size=1024,
        remove_columns=train_data.column_names,
        desc="Tokenizing train",
    )
    
    eval_data = eval_data.map(
        tokenize,
        batched=True,
        remove_columns=eval_data.column_names,
        desc="Tokenizing eval",
    )
    
    # Data collator for MLM
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    eval_loader = DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Train batches: {len(train_loader):,}")
    print(f"✓ Eval batches: {len(eval_loader):,}")
    
    # Estimate tokens
    total_tokens = len(train_data) * max_length / 1e6
    print(f"✓ Approx train tokens: {total_tokens:.1f}M")
    
    return train_loader, eval_loader