# Jetson Orin NX deployment notes

## Support boundary

The initial support target is JetPack 6.2.1 / L4T R36.4.7, CUDA 12.6, Ubuntu 22.04 and Python 3.10 on a Jetson Orin NX. The CLB carrier board may require a vendor-specific BSP or flashing procedure; flash and validate JetPack before using this repository.

## Validation sequence

1. Run `./scripts/bootstrap_upstream.sh`.
2. Run `./scripts/install_jetson.sh`.
3. Run `./scripts/verify_checkpoint.sh`.
4. Run `./scripts/verify_jetson.sh`.
5. Run `./scripts/infer_pair.sh` on one known rectified pair.

`verify_jetson.sh` checks the included checkpoint offline, strictly loads it, validates CUDA and executes a tiny GPU operation. It does not download MobileNetV2 weights.

## Common messages

| Message | Meaning | Action |
| --- | --- | --- |
| `xFormers not available` | Optional acceleration package is absent. | Ignore for LightStereo-S. |
| `CUDA is unavailable` | JetPack/PyTorch GPU setup is not usable. | Confirm JetPack version, rerun the installer, then inspect `nvidia-smi`/`tegrastats`. |
| checkpoint checksum failed | The included file was modified or truncated. | Restore it from Git, or deliberately use `verify_checkpoint.sh --recover`. |
| image sizes differ | Stereo pair is not geometrically compatible. | Use rectified left/right images with identical pixel dimensions. |

## Physical-device acceptance record

Before describing this project as hardware-verified, record the output of:

```bash
cat /etc/nv_tegra_release
./.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
./scripts/verify_jetson.sh
./scripts/benchmark_pair.sh --left /path/to/left.png --right /path/to/right.png
```
