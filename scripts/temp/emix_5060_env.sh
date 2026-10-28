#!/usr/bin/env bash
# Source this to run RoboTwin collection in the emix env on RTX 5060 (sm_120).
# emix has torch 2.10+cu128 which supports Blackwell; we supply a CUDA 13.0
# toolchain (conda-installed) so curobo's JIT extensions build for compute_120,
# and conda's gcc 15.2 as the host compiler.
#
# Usage: source scripts/temp/emix_5060_env.sh && python data_collection/robotwin.py ... --episodes 300
PROJ=/home/rogerspyke/projects/EMIX/Embodied-Mixup
EB=/home/rogerspyke/miniforge3/envs/emix
export CUDA_HOME="$EB"
export PATH="$EB/bin:$PATH"
export CPATH="$EB/targets/x86_64-linux/include:$EB/include"
export LIBRARY_PATH="$EB/targets/x86_64-linux/lib:$EB/lib"
export LD_LIBRARY_PATH="$EB/targets/x86_64-linux/lib:$EB/lib"
export CXX="$EB/bin/x86_64-conda-linux-gnu-g++"
export CC="$EB/bin/x86_64-conda-linux-gnu-gcc"
export PYTHONPATH="$PROJ/third_party/RoboTwin/envs/curobo/src:$PROJ/third_party/RoboTwin"
export TORCH_CUDA_ARCH_LIST="12.0"
