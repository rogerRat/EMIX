"""GPU HDBSCAN sidecar for EmixIndex builds at 10M+ pair scale.

The CPU HDBSCAN backends (hdbscan library and sklearn) cannot finish
clustering tens of millions of midpoints in reasonable time.  This sidecar
runs the same algorithm on GPU memory via RAPIDS cuML:

1. The build stages scaled midpoints with ``EMIX_CLUSTER_STAGE_DIR`` (see
   ``_cluster_pair_arrays``).
2. Run this script inside the cuML venv (~/.venvs/emix-cuml):
       python scripts/gpu_hbscan.py --stage-dir <dir> --device 1
3. Re-run the build with ``EMIX_CLUSTER_LABELS_PATH=<dir>/labels.npy``.

The staged midpoints are already standardized (StandardScaler-equivalent), so
this script needs only cuML/cupy/numpy — no sklearn, no dataset access.

Run with:
    ~/.venvs/emix-cuml/bin/python scripts/gpu_hbscan.py --stage-dir data/emix_cluster_stage/<task>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=str, required=True,
                        help="Directory containing midpoints_scaled.npy + params.json")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    parser.add_argument("--double", action="store_true",
                        help="Cluster in float64 (more VRAM; float32 default)")
    args = parser.parse_args()

    import cupy
    import numpy as np
    from cuml.cluster import HDBSCAN

    stage_dir = Path(args.stage_dir)
    params = json.loads((stage_dir / "params.json").read_text())
    points = np.load(stage_dir / "midpoints_scaled.npy")
    if args.double:
        points = points.astype(np.float64)
    print(f"[gpu_hbscan] points={points.shape} dtype={points.dtype} params={params}", flush=True)

    cupy.cuda.Device(args.device).use()
    mempool = cupy.get_default_memory_pool()
    mempool.free_all_free()

    clusterer = HDBSCAN(
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=int(params["min_samples"]),
        cluster_selection_epsilon=float(params["cluster_selection_epsilon"]),
        cluster_selection_method=params["cluster_selection_method"],
        metric="euclidean",
    )
    started = time.time()
    labels = clusterer.fit_predict(points)
    if hasattr(labels, "get"):  # cupy -> numpy
        labels = labels.get()
    labels = np.asarray(labels, dtype=np.int64)
    elapsed = time.time() - started
    num_clusters = int(np.unique(labels[labels >= 0]).size)
    noise = int(np.count_nonzero(labels == -1))
    print(
        f"[gpu_hbscan] fit complete in {elapsed:.1f}s: {num_clusters} clusters, "
        f"{noise} noise of {labels.shape[0]} pairs",
        flush=True,
    )

    np.save(stage_dir / "labels.npy", labels)
    print(f"[gpu_hbscan] labels written to {stage_dir / 'labels.npy'}", flush=True)
    mempool.free_all_free()


if __name__ == "__main__":
    main()
