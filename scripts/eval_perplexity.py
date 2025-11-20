import argparse
import math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='runs/baby_pretrain')
    ap.add_argument('--data', default='data/examples/tiny_corpus.txt')
    ap.add_argument('--seq_len', type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.eval()

    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()
    enc = tok(text, return_tensors='pt')

    nlls = []
    for i in range(0, enc['input_ids'].shape[1] - args.seq_len, args.seq_len):
        x = enc['input_ids'][:, i:i+args.seq_len]
        with torch.no_grad():
            out = model(x, labels=x)
            nlls.append(out.loss.item())
    ppl = math.exp(sum(nlls)/len(nlls)) if nlls else float('inf')
    print('Perplexity:', ppl)


if __name__ == '__main__':
    main()
