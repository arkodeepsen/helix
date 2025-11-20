# Helix🧬 ~100M Llama-style Language Model

A complete, modern training stack for building small language models from scratch on **AMD GPUs with ROCm**. Optimized for AMD Radeon RX 7000 & 9000 series (7800XT, 7900GRE, 7900XT, 7900XTX, 9060XT, 9070, 9070XT). Includes tokenizer training, data preparation, pretraining, supervised fine-tuning (SFT), optional DPO, GGUF export, agent demo, and Hugging Face deployment.

## Quickstart

1) **Install dependencies**

```powershell
pip install -r requirements.txt
```

2) **Train tokenizer**

```powershell
python scripts/train_tokenizer.py --input data/examples/tiny_corpus.txt --vocab_size 32000
```

3) **Prepare data** (tokenize to dataset shards)

```powershell
python scripts/prepare_data.py --tokenizer tokenizer --infile data/examples/tiny_corpus.txt --outdir data/processed --seq_len 2048
```

4) **Pretrain** (~50k steps starter)

```powershell
python scripts/pretrain.py --model_config configs/helix_100m.json --dataset_glob data/processed/helix_train --outdir runs/helix_pretrain --train_cfg configs/train_pretrain.yaml --tokenizer tokenizer
```

5) **Supervised fine-tuning** (instruction + tool-calls)

```powershell
python scripts/sft.py --base runs/helix_pretrain --data data/examples/sft_code_toolcalls.jsonl --outdir runs/helix_sft --cfg configs/train_sft.yaml
```

6) **Optional: DPO** (to improve JSON formatting and response quality)

```powershell
python scripts/dpo.py --base runs/helix_sft --data data/examples/dpo_pairs.jsonl --out runs/helix_dpo
```

7) **Export to GGUF** (for llama.cpp compatibility)

```powershell
python scripts/export_gguf.py --model_dir runs/helix_sft --llama_cpp_path ..\llama.cpp --out exports\helix-100m.gguf
# then quantize (from llama.cpp folder):
# .\quantize.exe ..\babyshark\exports\helix-100m.gguf ..\babyshark\exports\helix-100m.Q4_K_M.gguf Q4_K_M
```

8) **Test agent** (tool calling demo)

```powershell
python scripts/run_agent.py --model runs/helix_sft
# try prompts like: "Add 2 and 3 using a tool." or "Get weather for Delhi."
```

9) **Publish to Hugging Face**

```powershell
python scripts/push_to_hub.py --model_dir runs/helix_sft --repo_id <yourname>/helix-100m
```

## Project Structure

```
babyshark/
├─ README.md
├─ requirements.txt
├─ configs/              # Model and training configurations
│  ├─ helix_100m.json
│  ├─ train_pretrain.yaml
│  └─ train_sft.yaml
├─ data/
│  ├─ raw/              # Place your raw text files here
│  ├─ processed/        # Tokenized datasets
│  └─ examples/         # Example data
├─ runs/                # Training outputs
├─ scripts/             # Training and evaluation scripts
│  ├─ train_tokenizer.py
│  ├─ prepare_data.py
│  ├─ pretrain.py
│  ├─ sft.py
│  ├─ dpo.py
│  ├─ export_gguf.py
│  ├─ run_agent.py
│  ├─ push_to_hub.py
│  ├─ eval_perplexity.py
│  └─ eval_codegen.py
└─ tokenizer/           # Trained tokenizer files
```

## Hardware Requirements

**Supported AMD GPUs (ROCm):**
- AMD Radeon RX 9060XT
- AMD Radeon RX 9070 / 9070XT
- AMD Radeon RX 7800XT (16GB - recommended minimum)
- AMD Radeon RX 7900GRE
- AMD Radeon RX 7900XT / 7900XTX

**Minimum Requirements:**
- 16GB VRAM (e.g., RX 7800XT)
- ROCm 5.5 or later
- 32GB System RAM recommended

## Notes

- **ROCm Setup**: Install ROCm-compatible PyTorch from [pytorch.org](https://pytorch.org). See `docs/rocm.md` for detailed setup instructions.
- **Configuration**: All training configs are in `configs/` and can be easily customized for your GPU.
- **Data**: The example data is intentionally small for testing. Replace with higher-quality corpora for production training.
- **Performance**: Expect ~15-20k tokens/sec on RX 7900XTX, ~8-12k tokens/sec on RX 7800XT.

## Troubleshooting

- **GPU errors**: If you see CUDA/ROCm device errors, try adding `--bf16 false` in configs or switch to CPU for testing.
- **Memory issues**: Increase `gradient_accumulation_steps` and reduce `per_device_train_batch_size` in the config files.
- **Tokenization speed**: Pre-shard multiple files and increase `num_workers` in the data preparation script.
