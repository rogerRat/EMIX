"""Run the LeRobot policy in a separate Conda environment during evaluation."""

import atexit
import os
import socket
import subprocess
import sys
import tempfile
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _pack_tensor(value):
    array = value.detach().cpu().contiguous().numpy()
    return (array.dtype.str, array.shape, array.tobytes())


def _unpack_array(value):
    dtype, shape, data = value
    return np.frombuffer(data, dtype=np.dtype(dtype)).copy().reshape(shape)


def _rng_state(device):
    cpu = _pack_tensor(torch.get_rng_state())
    gpu = _pack_tensor(torch.cuda.get_rng_state(device)) if str(device).startswith("cuda") else None
    return cpu, gpu


def _set_rng_state(state, device):
    cpu, gpu = state
    torch.set_rng_state(torch.from_numpy(_unpack_array(cpu)))
    if gpu is not None:
        torch.cuda.set_rng_state(torch.from_numpy(_unpack_array(gpu)), device=device)


class IsolatedPolicy:
    def __init__(self, checkpoint, dataset_root, repo_id, device, env_name):
        self.device = device
        self.tmp = tempfile.TemporaryDirectory(prefix="emix-policy-")
        address = str(Path(self.tmp.name) / "socket")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(address)
        listener.listen(1)
        listener.settimeout(300)
        env = os.environ.copy()
        for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "EMIX_POLICY_WORKER_ENV"):
            env.pop(key, None)
        conda = env.get("CONDA_EXE", "/root/miniconda3/bin/conda")
        command = [conda, "run", "--no-capture-output",
                   "-p" if Path(env_name).is_absolute() else "-n", env_name, "python", "-u", "-m",
                   "evaluation.isolated_policy", address, str(checkpoint), str(dataset_root), repo_id, str(device)]
        self.proc = subprocess.Popen(command, cwd=ROOT, env=env)
        self.conn = None
        try:
            peer, _ = listener.accept()
            self.conn = Connection(peer.detach())
            ready = self._receive()
            self.shape = tuple(ready["shape"])
            self.policy_type = ready["policy_type"]
        except BaseException:
            self.close()
            raise
        finally:
            listener.close()
        atexit.register(self.close)

    def _receive(self):
        if not self.conn.poll(300):
            raise TimeoutError("Isolated policy did not respond within 300 seconds")
        message = self.conn.recv()
        if "error" in message:
            raise RuntimeError("Isolated policy failed:\n" + message["error"])
        return message

    def reset(self):
        self.conn.send({"op": "reset", "rng": _rng_state(self.device)})
        result = self._receive()
        _set_rng_state(result["rng"], self.device)

    def action(self, obs):
        self.conn.send({"op": "action",
                        "obs": {key: _pack_tensor(value) for key, value in obs.items()
                                if isinstance(value, torch.Tensor)},
                        "extra": {key: value for key, value in obs.items()
                                  if not isinstance(value, torch.Tensor)},
                        "rng": _rng_state(self.device)})
        result = self._receive()
        _set_rng_state(result["rng"], self.device)
        return _unpack_array(result["action"])

    def close(self):
        conn = getattr(self, "conn", None)
        if conn is not None:
            try:
                conn.send({"op": "close"})
                conn.close()
            except (OSError, EOFError):
                pass
            self.conn = None
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        tmp = getattr(self, "tmp", None)
        if tmp is not None:
            tmp.cleanup()


def load_isolated(checkpoint, dataset_root, repo_id, device, env_name):
    policy = IsolatedPolicy(checkpoint, dataset_root, repo_id, device, env_name)
    config = SimpleNamespace(type=policy.policy_type,
        input_features={"observation.images.head_camera": SimpleNamespace(shape=policy.shape)})
    return policy, None, None, config


def worker(address, checkpoint, dataset_root, repo_id, device):
    os.environ.pop("EMIX_POLICY_WORKER_ENV", None)
    from evaluation.evaluate import _load_policy, _policy_action
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(address)
    conn = Connection(sock.detach())
    try:
        policy, pre, post, config = _load_policy(checkpoint, dataset_root, repo_id, device)
        conn.send({"shape": tuple(config.input_features["observation.images.head_camera"].shape),
                   "policy_type": config.type})
        while True:
            request = conn.recv()
            if request["op"] == "close":
                break
            if request["op"] == "reset":
                _set_rng_state(request["rng"], device)
                policy.reset()
                conn.send({"rng": _rng_state(device)})
            elif request["op"] == "action":
                _set_rng_state(request["rng"], device)
                obs = {key: torch.from_numpy(_unpack_array(value)) for key, value in request["obs"].items()}
                obs.update(request.get("extra", {}))
                action = _policy_action(policy, pre, post, obs, device)
                conn.send({"action": _pack_tensor(torch.from_numpy(action)), "rng": _rng_state(device)})
            else:
                raise ValueError("Unknown isolated policy operation")
    except EOFError:
        pass
    except BaseException:
        try:
            conn.send({"error": traceback.format_exc()})
        except (OSError, EOFError):
            pass
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    worker(*sys.argv[1:])
