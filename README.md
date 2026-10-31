# LightStereo-S on Jetson Orin NX

Native, offline-weight inference overlay for **Jetson Orin NX** on **JetPack 6.2.1 / L4T R36.4.7**. It uses OpenStereo v2 commit `23d71c9` and ships the official 14.2 MB LightStereo-S-KITTI checkpoint, so the Jetson does not need to download model weights during setup or inference.

## What this repository is

- A reproducible Jetson overlay, not an OpenStereo source mirror.
- A pinned upstream clone plus a small patch that disables unrelated optional models and prevents MobileNetV2 from downloading generic pretrained weights.
- An academic/non-commercial-use workflow. Read [UPSTREAM.md](UPSTREAM.md) and [LICENSES.md](LICENSES.md) before use.

## Requirements

- Jetson Orin NX (ARM64, GPU compute capability 8.7)
- JetPack 6.2.1 / L4T R36.4.7, Ubuntu 22.04, Python 3.10
- Internet access only for the initial OpenStereo source clone and Python packages; **not** for model weights
- At least 4 GB free storage beyond the JetPack base installation

The scripts intentionally stop on a non-R36.4 L4T release rather than installing a possibly incompatible CUDA/PyTorch combination.

## Quick start

```bash
git clone https://github.com/Hbn040223/openstereo-lightstereo-jetson.git
cd openstereo-lightstereo-jetson

./scripts/bootstrap_upstream.sh
./scripts/install_jetson.sh
./scripts/verify_jetson.sh
```

Verify the bundled checkpoint at any time:

```bash
./scripts/verify_checkpoint.sh
```

Run one rectified stereo pair:

```bash
./scripts/infer_pair.sh \
  --left /path/to/left.png \
  --right /path/to/right.png \
  --output outputs/disparity.png \
  --gpu 0
```

The output is a color disparity visualisation PNG. It is not a raw, metric-depth or raw-disparity export. Left and right inputs must be rectified RGB images with identical dimensions.

## Benchmark

```bash
./scripts/benchmark_pair.sh --left /path/to/left.png --right /path/to/right.png --gpu 0
```

The benchmark performs one warm-up and reports five synchronized model forwards. It excludes process startup and output-file writing.

## Included checkpoint

`checkpoints/LightStereo-S-KITTI.ckpt` is included so an offline Jetson can use it immediately. Its required SHA-256 is:

```text
3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a
```

If you intentionally delete it, recovery is opt-in:

```bash
./scripts/verify_checkpoint.sh --recover
```

## Notes

- `xFormers not available` is an optional-feature notice and does not block LightStereo-S inference.
- This repository does not contain KITTI2015, any other dataset, cached Python packages, outputs, or credentials.
- See [docs/JETSON_ORIN_NX.md](docs/JETSON_ORIN_NX.md) for diagnostics and the physical-device acceptance checklist.
