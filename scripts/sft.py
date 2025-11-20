import json
import argparse
import yaml
import multiprocessing as mp
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='runs/helix_pretrain')
    ap.add_argument('--data', default='data/examples/sft_code_toolcalls.jsonl')
    ap.add_argument('--outdir', default='runs/helix_sft')
    ap.add_argument('--cfg', default='configs/train_sft.yaml')
    args = ap.parse_args()

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model = LlamaForCausalLM.from_pretrained(args.base)
    try:
        n_threads = max(1, mp.cpu_count() - 0)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            model = torch.compile(model, backend="inductor", mode="max-autotune")
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tok))

    raw = load_dataset('json', data_files=args.data, split='train')

    def fmt(ex):
        sys = "You are Helix, a helpful coding assistant."
        inst = ex.get('instruction', '')
        inp = ex.get('input', '')
        prompt = f"<s>[SYSTEM]\n{sys}\n[/SYSTEM]\n[INSTRUCTION]\n{inst}\n[/INSTRUCTION]\n"
        if inp:
            prompt += f"[INPUT]\n{inp}\n[/INPUT]\n"
        out = ex.get('output', '')
        text = prompt + out + tok.eos_token
        enc = tok(text)
        return {"input_ids": enc['input_ids'], "attention_mask": enc['attention_mask']}

    proc = raw.map(fmt, remove_columns=raw.column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    has_cuda = torch.cuda.is_available()
    use_bf16 = cfg.get('bf16', True) and has_cuda and torch.cuda.is_bf16_supported()

    args_t = TrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=cfg.get('per_device_train_batch_size', 4),
        gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 8),
        learning_rate=cfg.get('learning_rate', 1e-5),
        weight_decay=cfg.get('weight_decay', 0.0),
        warmup_steps=cfg.get('warmup_steps', 50),
        max_steps=cfg.get('total_steps', 2000),
        bf16=use_bf16,
        fp16=cfg.get('fp16', False) and has_cuda,
        no_cuda=not has_cuda,
        gradient_checkpointing=cfg.get('gradient_checkpointing', True),
        logging_steps=cfg.get('logging_steps', 20),
        save_steps=cfg.get('save_steps', 200),
    save_total_limit=cfg.get('save_total_limit', 2),
    remove_unused_columns=False,
    dataloader_pin_memory=False,
        report_to=[]
    )

    trainer = Trainer(model=model, args=args_t, train_dataset=proc, data_collator=collator)
    trainer.train()
    trainer.save_model(args.outdir)
    tok.save_pretrained(args.outdir)
    print('Saved SFT model to', args.outdir)


if __name__ == '__main__':
    main()
