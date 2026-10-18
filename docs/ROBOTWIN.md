# RoboTwin first-round protocol

The active benchmark is RoboTwin. Its fixed source is tracked directly under
`third_party/RoboTwin`; it is not a Git submodule. Use `emix` for conversion,
EMIX indexing and training, and `emix-twin` for the simulator. From the project
root, install each environment with its own entry point:

```bash
bash dependencies/_install.sh
bash dependencies/_install_twin.sh
bash dependencies/_install_twin.sh --check
```

On a headless server with NVIDIA graphics exposed, the twin installer first
repairs and verifies Vulkan/EGL and SAPIEN through `dependencies/_instal_sapien.sh`.
It then installs the CUDA runtime and Curobo. On Blackwell GPUs it builds OIDN
2.4.1 for the visible GPU architecture because SAPIEN's bundled OIDN 2.0.1
does not support `sm_120`. Keep `/usr/local/cuda` out of a Blackwell runtime's
library path when it contains an older CUDA build. The installer reuses
existing Conda environments; simulator assets and policy checkpoints are
separate. See [SOP-00](sop/SOP-00-environment-installation.md) for installation
and [the measured repair log](20261019_emix_sapien_runtime_repair.md) for the
RTX 5090 Vulkan/SAPIEN acceptance result.

The first experiment is `click_bell_left`: a dual-Piper scene where the left
Piper is the sole task actor and the right Piper stays at its home qpos. The
only policy image is `head_camera`; the policy state/action are the complete
two-Piper qpos vectors. Right-arm action labels are clamped to its initial home
qpos. The left EEF pose and left gripper signal are index-only features and are
not policy inputs.

The Piper separation is an explicit first-round configuration value of `0.60`
meters. The official Piper asset defines the individual robot pose but does not
publish a two-Piper separation value; the first collection smoke test must
validate this value. Use exactly the same value with `scripts/robotwin_eval.sh`.

Run the closed loop in order:

```bash
scripts/robotwin_collect.sh
scripts/robotwin_convert.sh
scripts/robotwin_train.sh
scripts/robotwin_eval.sh <piper-distance-m> <baseline-checkpoint> <emix-checkpoint>
```

The first protocol collects 200 expert episodes, trains baseline and EMIX
Diffusion Policy for 30,000 steps, and evaluates each checkpoint for 100 valid
RoboTwin seeds. EMIX mixes only `observation.images.head_camera`; state remains
the anchor state and EMIX batch probability is 30 percent.

ManiSkill-only source code and configuration are preserved under
`legacy/maniskill/`; the pre-migration backup is under
`backups/maniskill_migration_20260715045826/`.
