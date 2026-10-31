# Upstream and model attribution

This repository is a Jetson-specific overlay for [XiandaGuo/OpenStereo](https://github.com/XiandaGuo/OpenStereo), pinned to v2 commit `23d71c9`. It does not redistribute the OpenStereo source tree; `scripts/bootstrap_upstream.sh` clones that commit and applies this repository's patch.

The upstream README states that its code is for academic purposes and must not be used commercially. Users are responsible for satisfying all upstream terms.

`checkpoints/LightStereo-S-KITTI.ckpt` is the official LightStereo-S KITTI checkpoint from [XiandaGuo/OpenStereo on Hugging Face](https://huggingface.co/XiandaGuo/OpenStereo/blob/main/checkpoint/LightStereo/LightStereo-S-KITTI.ckpt). Its SHA-256 is recorded in `checkpoints/SHA256SUMS`. It is included with the user's authorization for reproducible academic use.
