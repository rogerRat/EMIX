"""Prepare, validate, and run three multi-task SmolVLA training arms."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

from embodied_mixup.policies.utils.multi_dataset import validate_sources
from embodied_mixup.policies.utils.multi_index import write_manifest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/scripts/multi_v0.yaml"
GENERATED = ROOT / "configs/scripts/multi_v0"
MANIFEST = ROOT / "data/multi-v0/emix_index/manifest.json"
ARMS = ("smolvla-baseline", "emix-mix", "emix-patch")


def load_spec():
    payload = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    tasks = payload["sources"]["tasks"]
    if len(tasks) != 10 or len(set(tasks)) != 10 or any(not value for value in tasks.values()):
        raise ValueError("multi-v0 requires ten distinct tasks with evaluation prompts")
    return payload


def make_training_config(spec, arm):
    training = spec["training"]
    repo_ids = list(spec["sources"]["tasks"])
    policy_type = "smolvla" if arm == "smolvla-baseline" else "emix_smolvla"
    policy = {"type": policy_type, "push_to_hub": False,
              **{key: training[key] for key in (
                  "pretrained_path", "device", "optimizer_lr", "n_obs_steps",
                  "chunk_size", "n_action_steps", "freeze_vision_encoder",
                  "train_expert_only", "train_state_proj", "load_vlm_weights",
                  "prefix_length", "pad_language_to", "num_expert_layers")}}
    payload = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "steps": training["steps"],
        "save_freq": training["save_freq"],
        "log_freq": training["log_freq"],
        "output_dir": str(ROOT / "outputs/multi-v0/train" / arm),
        "policy": policy,
        "dataset": {"root": spec["sources"]["dataset_root"],
                    "repo_id": repo_ids[0], "repo_ids": repo_ids,
                    "video_backend": training["video_backend"]},
        "protocol": {**spec["protocol"], "arm": arm},
    }
    if arm != "smolvla-baseline":
        emix = spec["emix"]
        payload["emix"] = {
            "index_path": str(MANIFEST),
            "mixup": {"alpha_mode": emix["alpha_mode"], "beta": emix["beta"],
                      "loss_weight": emix["loss_weight"],
                      "state_mode": emix["state_mode"],
                      "image_mode": "patch" if arm == "emix-patch" else "linear",
                      "annotation_root": spec["sources"]["dataset_root"],
                      "detect_overlap": emix["detect_overlap"]},
            "training": {"enabled": True, "training_mode": "mixed",
                         "seed": spec["protocol"]["training_seed"],
                         "scheduler": emix["scheduler"]},
        }
    return payload


def make_eval_config(spec, arm, task, prompt):
    training = spec["training"]
    evaluation = spec["evaluation"]
    checkpoint = ROOT / "outputs/multi-v0/train" / arm / "checkpoints" / "{:06d}".format(training["steps"]) / "pretrained_model"
    return {
        "dataset": {"repo_id": task, "root": str(Path(spec["sources"]["dataset_root"]) / task)},
        "evaluation": {
            "checkpoint": str(checkpoint),
            "output_dir": str(ROOT / "outputs/multi-v0/eval" / arm / task),
            "benchmark_root": str(ROOT / "third_party/RoboTwin"),
            "task_name": task, "task_config": task,
            "task_prompt": prompt, "piper_distance": evaluation["piper_distance"],
            "episodes": evaluation["episodes"], "seed": evaluation["seed"],
            "device": evaluation["device"], "save_video": evaluation["save_video"],
            "video_fps": evaluation["video_fps"],
        },
        "protocol": {**spec["protocol"], "arm": arm, "task": task},
    }


def prepare(spec):
    source_root = Path(spec["sources"]["dataset_root"])
    repo_ids = list(spec["sources"]["tasks"])
    validate_sources(source_root, repo_ids)
    write_manifest(MANIFEST, repo_ids, source_root, Path(spec["sources"]["index_root"]))
    for arm in ARMS:
        arm_dir = GENERATED / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "train.yaml").write_text(
            yaml.safe_dump(make_training_config(spec, arm), sort_keys=False), encoding="utf-8")
        for task, prompt in spec["sources"]["tasks"].items():
            (arm_dir / f"eval_{task}.yaml").write_text(
                yaml.safe_dump(make_eval_config(spec, arm, task, prompt), sort_keys=False),
                encoding="utf-8")
    cross_pair_count = json.loads(MANIFEST.read_text(encoding="utf-8"))["cross_pair_count"]
    print(f"Prepared {len(ARMS)} training and {len(ARMS) * len(repo_ids)} evaluation configs; cross-task pairs: {cross_pair_count}")


def mock(spec):
    from embodied_mixup.policies.train import _build_train_config

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = list(spec["sources"]["tasks"])
    if [item["repo_id"] for item in manifest["sources"]] != ids:
        raise AssertionError("Manifest source order mismatch")
    cross_index = json.loads(Path(manifest["cross_pairs_path"]).read_text(encoding="utf-8"))
    pairs = cross_index["pairs"]
    if (len(pairs) != manifest["cross_pair_count"]
            or {pair["a"]["repo_id"] for pair in pairs} != set(ids)
            or {pair["b"]["repo_id"] for pair in pairs} != set(ids)):
        raise AssertionError("Cross-task index does not cover all ten anchor and donor tasks")
    configs = []
    for arm in ARMS:
        path = GENERATED / arm / "train.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = _build_train_config(payload, path)
        if config.dataset_repo_ids != ids or config.policy_cfg.pretrained_path is None:
            raise AssertionError(f"Invalid training config for {arm}")
        configs.append(config)
        for task in ids:
            eval_payload = yaml.safe_load((GENERATED / arm / f"eval_{task}.yaml").read_text())
            if eval_payload["evaluation"]["task_prompt"] != spec["sources"]["tasks"][task]:
                raise AssertionError(f"Evaluation prompt mismatch for {arm}/{task}")
    common = [(cfg.batch_size, cfg.steps, cfg.num_workers, cfg.policy_cfg.chunk_size,
               cfg.policy_cfg.n_action_steps, str(cfg.policy_cfg.pretrained_path)) for cfg in configs]
    if len(set(common)) != 1:
        raise AssertionError("Training arms differ outside their treatment")
    from embodied_mixup.policies.utils.batch import get_policy_temporal_spec, stack_tensor_list
    from embodied_mixup.policies.utils.emix_loader_registry import make_emix_loader
    from embodied_mixup.policies.utils.emix_loader_registry_utils import EmixLoaderSettings
    from embodied_mixup.policies.utils.multi_cross_loader import CrossTaskEmixDataset
    from embodied_mixup.policies.utils.multi_dataset import load_multi_dataset

    baseline_meta, baseline_dataset = load_multi_dataset(
        spec["sources"]["dataset_root"], ids, configs[0].policy_cfg,
        spec["training"]["video_backend"])
    baseline_sample = baseline_dataset[0]
    if not isinstance(baseline_sample["task"], str) or tuple(baseline_sample["action"].shape) != (50, 7):
        raise AssertionError("Real multi-task baseline sample is invalid")
    temporal = get_policy_temporal_spec(configs[2].policy_cfg)
    source = manifest["sources"][0]
    patch_settings = EmixLoaderSettings(
        policy_type="emix_smolvla", mixup_image_mode="patch", video_backend="torchcodec",
        annotation_root=source["annotation_root"])
    within = make_emix_loader(
        "emix_smolvla", source["root"], source["repo_id"],
        source["index_path"], temporal, patch_settings)[0]
    for auxiliary_key in ("observation.eef_pose", "observation.left_gripper", "emix_eef_poses"):
        within.pop(auxiliary_key, None)
    for mode in ("linear", "patch"):
        cross = CrossTaskEmixDataset(
            manifest["cross_pairs_path"], manifest["sources"], temporal,
            EmixLoaderSettings(policy_type="emix_smolvla", mixup_image_mode=mode,
                               video_backend="torchcodec"))
        item = cross[0]
        pair = cross.pairs[0]
        if pair["a"]["repo_id"] == pair["b"]["repo_id"] or not item["task"]:
            raise AssertionError("Cross-task provenance or anchor task is invalid")
        if tuple(item["action"].shape) != (50, 7) or tuple(item["emix_actions"][1].shape) != (50, 7):
            raise AssertionError("Cross-task action windows are invalid")
        if mode == "patch":
            import torch
            image_key = "observation.images.head_camera"
            source_a = item["emix_source_images"][0][image_key][0]
            source_b = item["emix_source_images"][1][image_key][0]
            mixed_image = item[image_key][0]
            mask = torch.as_tensor(cross.annotations[pair["b"]["repo_id"]].target_mask(
                pair["b"]["frame_index"]), dtype=torch.bool)
            if (not mask.any() or not (~mask).any()
                    or not torch.equal(mixed_image[:, mask], source_b[:, mask])
                    or not torch.equal(mixed_image[:, ~mask], source_a[:, ~mask])):
                raise AssertionError("Cross-task patch mask or source image is wrong")
            if set(item) != set(within):
                raise AssertionError(f"Mixed batch keys differ: {set(item) ^ set(within)}")
            stack_tensor_list([within, item])
            stack_tensor_list([item, within])
        else:
            if "emix_alpha" not in item:
                raise AssertionError("Linear cross-task augmentation lacks alpha")
    print(f"Mock validated 3 training configs, 30 evaluation configs, "
          f"{len(ids)} sources, {manifest['cross_pair_count']} cross-task pairs, "
          "and real baseline/mix/patch samples")


def completed_eval(path, checkpoint, episodes):
    if not path.is_dir():
        return False
    for metrics in path.glob("*/metrics.json"):
        try:
            payload = json.loads(metrics.read_text(encoding="utf-8"))
            if payload.get("episodes") == episodes and Path(payload["checkpoint"]).resolve() == checkpoint.resolve():
                return True
        except (OSError, ValueError, KeyError):
            continue
    return False


def run(spec):
    conda = "/root/miniconda3/bin/conda"
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    env["EMIX_POLICY_WORKER_ENV"] = "/root/autodl-tmp/conda-envs/emix-reinstall"
    for arm in ARMS:
        train_config = GENERATED / arm / "train.yaml"
        checkpoint_root = ROOT / "outputs/multi-v0/train" / arm / "checkpoints"
        final_model = checkpoint_root / "{:06d}".format(spec["training"]["steps"]) / "pretrained_model/model.safetensors"
        if not final_model.is_file():
            command = [conda, "run", "--no-capture-output", "-p", "/root/autodl-tmp/conda-envs/emix-reinstall",
                       "python", "-u", "train/train.py", "--config", str(train_config),
                       "--seed", str(spec["protocol"]["training_seed"])]
            last = checkpoint_root / "last"
            if last.is_symlink():
                command.extend(["--resume_from", str(last.resolve())])
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        for task in spec["sources"]["tasks"]:
            output_dir = ROOT / "outputs/multi-v0/eval" / arm / task
            if completed_eval(output_dir, final_model.parent, spec["evaluation"]["episodes"]):
                continue
            command = [conda, "run", "--no-capture-output", "-p", "/root/autodl-tmp/conda-envs/emix-twin-reinstall",
                       "python", "-u", "evaluation/evaluate.py", "--config",
                       str(GENERATED / arm / f"eval_{task}.yaml")]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    marker = ROOT / "outputs/multi-v0/.complete"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("3 training arms and 30 task evaluations complete\n", encoding="utf-8")
    print("multi-v0 complete: 3 training arms and 30 task evaluations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "mock", "run"), required=True)
    args = parser.parse_args()
    spec = load_spec()
    if args.mode == "prepare":
        prepare(spec)
    elif args.mode == "mock":
        mock(spec)
    else:
        prepare(spec)
        mock(spec)
        run(spec)


if __name__ == "__main__":
    main()
