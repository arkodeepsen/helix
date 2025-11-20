# ROCm notes (AMD 7800XT)

- ROCm support is strongest on Linux. On Windows, ROCm is not generally supported. Prefer Linux/WSL for GPU training with 7800XT.
- Install the PyTorch ROCm build from pytorch.org (select ROCm channel matching your driver). Verify with:

```bash
python -c "import torch; print(torch.version.__version__, torch.version.hip, torch.cuda.is_available(), torch.backends.mps.is_available())"
```

- If `bitsandbytes` fails to install on ROCm, it's optional; scripts don't require it.
- Use `bf16` where possible; otherwise set `bf16: false` in configs and fall back to `fp16`.
- If VRAM is tight, increase `gradient_accumulation_steps`, reduce `per_device_train_batch_size`, and enable `gradient_checkpointing` (already on in configs).
