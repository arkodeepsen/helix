#!/usr/bin/env python3
"""
Train a tokenizer using HuggingFace tokenizers (no sentencepiece dependency)
"""
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def train_tokenizer(input_file, vocab_size=50000, model_prefix="babyshark_bpe"):
    """Train a BPE tokenizer using HuggingFace tokenizers"""
    
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # Set up the trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<unk>", "<pad>", "<bos>", "<eos>", 
            "</n>", "<tab>", "<space>", 
            "<json>", "</json>", "<code>", "</code>"
        ],
        min_frequency=2,
        continuing_subword_prefix="",
        end_of_word_suffix="</w>"
    )
    
    # Use whitespace pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Train the tokenizer
    print(f"Training tokenizer on {input_file}...")
    tokenizer.train([input_file], trainer)
    
    # Add post-processor for proper token handling
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    
    # Save the tokenizer
    tokenizer.save(f"{model_prefix}.json")
    
    # Also create a HuggingFace compatible tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    
    # Save HuggingFace format
    hf_tokenizer.save_pretrained(f"{model_prefix}_hf")
    
    print(f"✅ Tokenizer saved to:")
    print(f"  - {model_prefix}.json (tokenizers format)")
    print(f"  - {model_prefix}_hf/ (HuggingFace format)")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Test the tokenizer
    test_text = "Hello world! This is a test of the Helix tokenizer."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\nTest encoding:")
    print(f"Original: {test_text}")
    print(f"Tokens: {encoded.tokens}")
    print(f"Decoded: {decoded}")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer with HuggingFace tokenizers")
    parser.add_argument('--input', required=True, help='Input text file for training')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--model_prefix', default='helix_tokenizer', help='Model prefix for output files')
    
    args = parser.parse_args()
    
    train_tokenizer(args.input, args.vocab_size, args.model_prefix)


if __name__ == '__main__':
    main()