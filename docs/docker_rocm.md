# Train on AMD GPU via Docker + WSL2 (ROCm)

Requirements
- Windows 11, WSL2, Ubuntu 22.04 distro installed
- Docker Desktop with WSL2 backend enabled
- AMD 7000-series GPU (e.g., 7800XT)

Steps
1) Ensure /dev/kfd and /dev/dri are exposed in your WSL2 Ubuntu (install AMD drivers/ROCm runtime in WSL2 per AMD docs).
2) From Windows PowerShell (project root), build the image:

```powershell
docker compose -f docker\compose.rocm.yml build
```

3) Verify GPU access in container:

```powershell
docker compose -f docker\compose.rocm.yml run --rm train bash -lc "python docker/verify_rocm.py"
```

4) Kick off a short GPU pretrain:

```powershell
docker compose -f docker\compose.rocm.yml up --remove-orphans
```

Notes
- For some 7xxx GPUs, `HSA_OVERRIDE_GFX_VERSION=11.0.0` helps expose compute.
- If GPU not visible, check that `ls /dev/kfd` and `ls /dev/dri` work inside your WSL2 Ubuntu, and your user is in `video`/`render` groups.
- You can tune `configs/train_pretrain.yaml` for longer runs once this short job is validated.
