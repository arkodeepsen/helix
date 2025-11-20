#!/usr/bin/env python3
"""
Prepare training data using our HuggingFace BabyShark tokenizer
"""
import os
import glob
import argparse
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizerFast


def load_babyshark_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """Load our custom BabyShark tokenizer"""
    print(f"Loading BabyShark tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Print tokenizer info
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    
    return tokenizer


def tokenize_and_chunk(tokenizer: PreTrainedTokenizerFast, text: str, seq_len: int):
    """Tokenize text and create fixed-length sequences - MEMORY EFFICIENT"""
    print(f"Processing {len(text):,} characters in chunks...")
    
    # Process in 10MB chunks to avoid memory explosion
    chunk_size = 10 * 1024 * 1024  # 10MB chunks
    examples = []
    total_tokens = 0
    
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(text) + chunk_size - 1)//chunk_size} ({len(chunk_text):,} chars)")
        
        # Tokenize this chunk
        encoding = tokenizer(
            chunk_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False
        )
        
        input_ids = encoding['input_ids']
        total_tokens += len(input_ids)
        
        # Create training sequences from this chunk
        seq_size = seq_len - 2  # Reserve space for BOS/EOS
        
        for j in range(0, len(input_ids), seq_size):
            sequence = input_ids[j:j + seq_size]
            
            # Add special tokens
            if tokenizer.bos_token_id is not None:
                sequence = [tokenizer.bos_token_id] + sequence
            if tokenizer.eos_token_id is not None:
                sequence = sequence + [tokenizer.eos_token_id]
            
            # Pad if needed
            if len(sequence) < seq_len:
                if tokenizer.pad_token_id is not None:
                    sequence = sequence + [tokenizer.pad_token_id] * (seq_len - len(sequence))
            
            # Create attention mask
            attention_mask = [1] * min(len(sequence), seq_len)
            if len(attention_mask) < seq_len:
                attention_mask = attention_mask + [0] * (seq_len - len(attention_mask))
            
            # Ensure exact length
            sequence = sequence[:seq_len]
            attention_mask = attention_mask[:seq_len]
            
            examples.append({
                "input_ids": sequence,
                "attention_mask": attention_mask,
                "labels": sequence.copy()
            })
        
        # Clear chunk from memory
        del encoding, input_ids
    
    print(f"Created {len(examples):,} training examples from {total_tokens:,} tokens")
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser(description="Prepare training data with BabyShark tokenizer")
    parser.add_argument('--tokenizer', default='tokenizer', help='Path to BabyShark tokenizer directory')
    parser.add_argument('--infile', required=True, help='Input text file or directory')
    parser.add_argument('--outdir', default='data/processed', help='Output directory')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--shard_size', type=int, default=10000, help='Examples per shard')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = load_babyshark_tokenizer(args.tokenizer)
    
    # Collect input files
    input_files = []
    if os.path.isdir(args.infile):
        for ext in ['*.txt', '*.md', '*.json']:
            input_files.extend(glob.glob(os.path.join(args.infile, '**', ext), recursive=True))
    else:
        input_files.append(args.infile)
    
    print(f"Found {len(input_files)} input files")
    
    # Process each file
    datasets = []
    total_chars = 0
    
    for file_path in input_files:
        try:
            print(f"\nProcessing: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            total_chars += len(text)
            
            if len(text.strip()) == 0:
                print(f"Skipping empty file: {file_path}")
                continue
                
            dataset = tokenize_and_chunk(tokenizer, text, args.seq_len)
            datasets.append(dataset)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not datasets:
        raise RuntimeError("No valid input data found!")
    
    # Combine all datasets
    print(f"\nCombining {len(datasets)} datasets...")
    combined_dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    
    print(f"\nHelix Dataset Statistics:")
    print(f"  Total characters processed: {total_chars:,}")
    print(f"  Total training examples: {len(combined_dataset):,}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Total tokens: {len(combined_dataset) * args.seq_len:,}")
    
    # Save dataset
    output_path = os.path.join(args.outdir, 'helix_train')
    combined_dataset.save_to_disk(output_path)
    print(f"\n✅ Dataset saved to: {output_path}")
    
    # Save a sample for inspection
    sample_path = os.path.join(args.outdir, 'sample.txt')
    with open(sample_path, 'w', encoding='utf-8') as f:
        for i in range(min(3, len(combined_dataset))):
            example = combined_dataset[i]
            decoded = tokenizer.decode(example['input_ids'], skip_special_tokens=False)
            f.write(f"=== Sample {i+1} ===\n")
            f.write(f"Input IDs: {example['input_ids'][:20]}...\n")
            f.write(f"Decoded: {decoded[:200]}...\n\n")
    
    print(f"Sample outputs saved to: {sample_path}")


if __name__ == '__main__':
    main()