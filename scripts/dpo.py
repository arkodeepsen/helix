import argparse
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments
from trl import DPOTrainer


def load_pref_dataset(path: str) -> Dataset:
    ds = load_dataset('json', data_files=path, split='train')
    # Expect fields: prompt, chosen, rejected
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='Base SFT model directory')
    ap.add_argument('--data', required=True, help='JSONL with prompt, chosen, rejected')
    ap.add_argument('--out', default='runs/helix_dpo')
    args = ap.parse_args()

    model = LlamaForCausalLM.from_pretrained(args.base)
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)

    ds = load_pref_dataset(args.data)

    has_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        max_steps=200,
        bf16=False if not has_cuda else True,
        no_cuda=not has_cuda,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        logging_steps=20,
        save_steps=200,
        report_to=[]
    )

    trainer = DPOTrainer(
        model,
        ref_model=None,
        beta=0.1,
        train_dataset=ds,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print('Saved DPO-tuned model to', args.out)


if __name__ == '__main__':
    main()
