import json
import os
import argparse
import yaml
import torch
import multiprocessing as mp
import platform
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    LlamaConfig, LlamaForCausalLM, LlamaTokenizer,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)


def load_tokenizer(path: str):
    try:
        # Try loading as HF tokenizer first (directory or json)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception:
        # Fallback to LlamaTokenizer for SentencePiece models
        tok = LlamaTokenizer(vocab_file=path)
    
    # Ensure special tokens exist
    special_tokens = {}
    if tok.pad_token is None:
        special_tokens["pad_token"] = "<pad>"
    if tok.bos_token is None:
        special_tokens["bos_token"] = "<s>"
    if tok.eos_token is None:
        special_tokens["eos_token"] = "</s>"
    if tok.unk_token is None:
        special_tokens["unk_token"] = "<unk>"
        
    if special_tokens:
        tok.add_special_tokens(special_tokens)
        
    return tok


def load_dataset_paths(pattern: str):
    if any(ch in pattern for ch in ['*', '?']):
        # glob for multiple saved datasets (directories)
        import glob
        paths = [p for p in glob.glob(pattern) if os.path.isdir(p)]
        if not paths:
            raise FileNotFoundError(f'No dataset dirs match pattern: {pattern}')
        dsets = [load_from_disk(p) for p in paths]
        return concatenate_datasets(dsets)
    else:
        return load_from_disk(pattern)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_config', default='configs/helix_100m.json')
    ap.add_argument('--dataset_glob', default='data/processed/helix_train')
    ap.add_argument('--outdir', default='runs/helix_pretrain')
    ap.add_argument('--train_cfg', default='configs/train_pretrain.yaml')
    ap.add_argument('--tokenizer', default='tokenizer', help='Path to tokenizer (HF dir or SP model)')
    args = ap.parse_args()

    # load model config and init



    # CPU threading optimizations
    try:
        n_threads = max(1, mp.cpu_count() - 0)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
    except Exception:
        pass

    # Optional compile (PyTorch 2.x): only when CUDA is available to avoid Windows CPU compiler issues
    try:
        if torch.cuda.is_available():
            model = torch.compile(model, backend="inductor", mode="max-autotune")
    except Exception:
        pass

    # Matmul precision hint for CPU BLAS
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # tokenizer
    tok = load_tokenizer(args.tokenizer)
    model.resize_token_embeddings(len(tok))

    # data
    ds = load_dataset_paths(args.dataset_glob)

    # collator
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # training args
    with open(args.train_cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Auto-fallback for CPU-only environments
    has_cuda = torch.cuda.is_available()
    use_bf16 = cfg.get('bf16', True) and has_cuda and torch.cuda.is_bf16_supported()

    ta = TrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=cfg.get('per_device_train_batch_size', 2),
        gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 32),
        learning_rate=cfg.get('learning_rate', 3e-4),
        weight_decay=cfg.get('weight_decay', 0.05),
        warmup_steps=cfg.get('warmup_steps', 1000),
        max_steps=cfg.get('total_steps', 50000),
        lr_scheduler_type=cfg.get('lr_scheduler_type', 'cosine'),
        bf16=use_bf16,
        fp16=cfg.get('fp16', False) and has_cuda,
        no_cuda=not has_cuda,
        gradient_checkpointing=cfg.get('gradient_checkpointing', True),
        max_grad_norm=cfg.get('max_grad_norm', 1.0),
    logging_steps=cfg.get('logging_steps', 50),
        save_steps=cfg.get('save_steps', 1000),
        save_total_limit=cfg.get('save_total_limit', 3),
        dataloader_num_workers=cfg.get('num_workers', 2),
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to=[]
    )

    trainer = Trainer(model=model, args=ta, train_dataset=ds, data_collator=collator)
    trainer.train()

    os.makedirs(args.outdir, exist_ok=True)
    trainer.save_model(args.outdir)
    tok.save_pretrained(args.outdir)
    print('Saved model to', args.outdir)


if __name__ == '__main__':
    main()
