# Initial workspace snapshot

Captured: 2026-09-29T23:18:23+08:00
Target repository: https://github.com/rogerRat/EMIX, branch `main`.

This is a new root commit containing the current working tree, including local
changes and uncommitted source files. It does not import the previous Git history.

- EMIX source repository: https://github.com/Koorye/Embodied-Mixup
- Source base commit: `0487c234ad5ccf7209a1de009e19cac2664e4d54` (plus the captured working-tree changes)
- RoboTwin source repository: https://github.com/RogersPyke/RoboTwin
- RoboTwin source commit: `2de39e730d6099851e4bb504eecaa57d33eec9c0`
- RoboTwin tracked source is included directly in `third_party/RoboTwin` rather
  than as a Git submodule. Its existing license files remain in place.

Included: source, configurations (including captured generated run configs),
all local tests, documentation and local diagnostic scripts. Existing ignore
rules are retained; snapshot files deliberately covered by those rules are
explicitly tracked in this initial commit.

Excluded: original Git metadata/history, datasets, downloaded simulator assets,
model checkpoints, training/evaluation output directories, logs, caches,
bytecode, virtual environments and credential files. Asset/environment setup
is still required to run simulation. No live process was stopped or modified.

`docs/_exp.md` and runtime configs represent their state at capture time;
running experiments continue updating the original workspace independently.

The instance annotation module has 11 passing CPU unit tests. This snapshot
also preserves pre-existing tests and code without claiming the entire suite
has passed.
