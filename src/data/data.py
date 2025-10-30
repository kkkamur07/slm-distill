"""Data loading and preprocessing for Sangraha dataset"""

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


def prepare_datasets(
    dataset_name: str,
    subset_name: str,
    data_percent: float,
    tokenizer,
    max_length: int,
    batch_size: int,
    num_workers: int = 2
):
    
    print(f"\nLoading Sangraha dataset...")
    print(f"Dataset: {dataset_name}")
    print(f"Subset: {subset_name}")
    print(f"Using: {data_percent}% of data")
    
    # Calculate splits
    train_split = f"hin[:{data_percent}%]"
    eval_split = f"hin[{data_percent}%:{data_percent + 0.2}%]"
    
    print(f"Train split: {train_split}")
    print(f"Eval split: {eval_split}")
    
    # Load datasets
    # 'name' parameter selects the configuration/subset
    train_data = load_dataset(
        path=dataset_name,       
        name=subset_name,        
        split=train_split,        
        trust_remote_code=True
    )
    
    eval_data = load_dataset(
        path=dataset_name,
        name=subset_name,
        split=eval_split,
        trust_remote_code=True
    )
    
    print(f"✓ Train: {len(train_data):,} examples")
    print(f"✓ Eval: {len(eval_data):,} examples")
    print(f"Columns: {train_data.column_names}")
    
    # Tokenization function
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )
    
    # Tokenize datasets
    print("Tokenizing...")
    train_data = train_data.map(
        tokenize,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Tokenizing train",
        num_proc=4
    )
    
    eval_data = eval_data.map(
        tokenize,
        batched=True,
        remove_columns=eval_data.column_names,
        desc="Tokenizing eval",
        num_proc=4
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
        pin_memory=True
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
