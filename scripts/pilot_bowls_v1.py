"""Run a reproducible bowls layout/expert pilot without starting training.

Example:
  ROBOTWIN_V1_BOWLS_WIDE_SPAWN=1 ROBOTWIN_V1_BOWLS_HALF_X=0.33 \\
      python scripts/pilot_bowls_v1.py --seeds 100 --start 8000
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'third_party' / 'RoboTwin'
TASK = 'stack_bowls_two_left_cen_arm_top_cam'


def build_task():
    module = importlib.import_module(f'envs.left.cen_arm_top_cam.{TASK}')
    task = getattr(module, TASK)()
    config_path = BENCHMARK / 'task_config' / 'left' / 'cen_arm_top_cam' / f'{TASK}.yml'
    config = yaml.safe_load(config_path.read_text())
    config.update(task_name=TASK, task_config=TASK, eval_mode=True,
                  left_robot_file=str(BENCHMARK / 'assets/embodiments/piper'),
                  right_robot_file=str(BENCHMARK / 'assets/embodiments/piper'),
                  dual_arm_embodied=False, embodiment_dis=0.60)
    robot_config = yaml.safe_load((BENCHMARK / 'assets/embodiments/piper/config.yml').read_text())
    config['left_embodiment_config'] = robot_config
    config['right_embodiment_config'] = robot_config
    return task, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=100)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--output', type=Path,
                        default=Path('results/pilot/v1_bowls/wide_spawn.pilot.jsonl'))
    args = parser.parse_args()
    if args.seeds <= 0 or args.start < 0:
        parser.error('seeds must be positive and start nonnegative')
    output = args.output.resolve()
    os.chdir(BENCHMARK)
    sys.path.insert(0, str(BENCHMARK))
    rows = []
    for seed in range(args.start, args.start + args.seeds):
        task, config = build_task()
        row = {'task': TASK, 'seed': seed,
               'spawn_half_x': os.environ.get('ROBOTWIN_V1_BOWLS_HALF_X', '0.33'),
               'wide_spawn': os.environ.get('ROBOTWIN_V1_BOWLS_WIDE_SPAWN', '1') == '1'}
        try:
            task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
            row['layout'] = task.sampled_layout_summary
            row['stack_target'] = list(map(float, task.stack_target_xy))
            task.play_once()
            row['plan_success'] = bool(task.plan_success)
            row['success'] = bool(task.check_success())
        except Exception as exc:
            row.update(plan_success=False, success=False,
                       error=f'{type(exc).__name__}: {exc}'[:200])
        finally:
            try:
                task.close_env()
            except Exception:
                pass
        rows.append(row)
        wins = sum(bool(item['success']) for item in rows)
        print(f'[pilot] seed={seed} success={row["success"]} '
              f'running={wins}/{len(rows)}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    print(f'[pilot] FINAL {sum(bool(row["success"]) for row in rows)}/{len(rows)}; {output}', flush=True)


if __name__ == '__main__':
    main()
