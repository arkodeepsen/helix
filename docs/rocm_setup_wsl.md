# Fast ROCm Setup for 7800XT in WSL

## Enable ROCm in WSL Ubuntu

Open your WSL Ubuntu terminal and run these commands:

```bash
# Add ROCm repository
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.1 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm runtime
sudo apt update
sudo apt install -y rocm-hip-runtime rocminfo

# Add your user to GPU groups
sudo usermod -aG render,video $USER

# Apply group changes
newgrp render
```

## Install PyTorch ROCm and dependencies

```bash
# Create Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch ROCm (optimized for 7800XT)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# Install training dependencies
pip install -r requirements.txt
```

## Test GPU

```bash
# Quick GPU verification
python - << 'PY'
import torch, time
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('HIP version:', getattr(torch.version, 'hip', 'Not found'))
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
    
    # Speed test
    x = torch.randn((4096, 4096), device='cuda', dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        x = x @ x
    torch.cuda.synchronize()
    print('GPU matmul time:', time.time() - t0, 'seconds')
else:
    print('GPU not detected - check ROCm setup')
PY
```

## Run GPU training

```bash
# Start 100M model pretrain on GPU (200 steps)
python scripts/pretrain.py \
  --model_config configs/model_baby_100m.json \
  --dataset_glob data/processed/shard_0000 \
  --outdir runs/baby_pretrain_gpu \
  --train_cfg configs/train_pretrain_rocm_short.yaml \
  --spm bullshark_bpe.model
```

Expected performance:
- GPU memory usage: ~8-12GB (you have 16GB, so plenty of headroom)
- Training speed: 10-50x faster than CPU
- 200 steps should complete in minutes instead of hours

If you get VRAM errors, reduce `per_device_train_batch_size` from 4 to 2 in the config.