# RoboTwin first-round protocol

The active benchmark is RoboTwin. Initialize its fixed Fork source with:

```bash
git submodule update --init --recursive third_party/RoboTwin
```

Use `emix` only for conversion, EMIX index construction, and training. Create
`emix-twin` with `scripts/create_emix_twin_env.sh`, then use it only for
RoboTwin assets, collection, and online evaluation. RoboTwin modifies installed
SAPIEN and MPLib files during its official setup, so it must not share `emix`.

The RTX 5060 uses the separate `emix-twin-5060` environment. SAPIEN 3.0.0b1
bundles OIDN 2.0.1, whose CUDA backend does not recognize Blackwell
`sm_120`. Do not change the renderer to `none`: that makes both GPUs emit
raw ray-tracing noise. Instead, activate the target environment and run
`bash scripts/install_robotwin_oidn.sh` with CUDA 12.8 available. The script
builds the pinned OIDN 2.4.1 tag with `OIDN_DEVICE_CPU=OFF`, `OIDN_DEVICE_CUDA=ON`,
`OIDN_FILTER_RT=ON`, and architecture `120`, then installs a Conda activation
hook for the CUDA/C++ runtime and copies the versioned libraries into
SAPIEN's private `oidn_library` directory. Run
`bash scripts/verify_robotwin_render.sh`
before collection or evaluation. It fails if CUDA is unavailable or a CPU OIDN
plugin is present. `configs/scripts/run.yaml` routes GPU 1 collection and
evaluation to this environment; keep `/usr/local/cuda` out of its runtime path.
For the RTX 4070, use the same script in its own RoboTwin environment with
`OIDN_ARCHITECTURES=89`; both environments remain GPU-only and use OIDN.

For a fresh machine, first run `third_party/RoboTwin/script/_install.sh` in the
dedicated environment, then run the two root scripts above. The source tag,
GPU-only CMake options, CUDA architecture, and smoke-test policy are kept in
the repository; the generated build tree remains outside Git. Existing Conda
environments are reused and never deleted by these scripts.

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
