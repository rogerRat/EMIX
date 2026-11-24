"""PI0.5 LoRA parameter search with 10k checkpoint evaluations."""
from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path
import yaml
from embodied_mixup.policies.utils.multi_dataset import validate_sources
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/"configs/scripts/v0_pi05_paramsearch.yaml"
GEN=ROOT/"configs/scripts/v0_pi05_paramsearch"
OUT=Path("/root/autodl-tmp/outputs/v0-pi05-paramsearch")
TRAIN=OUT/"train/pi05-lora"
LOG=OUT/"logs"
ENV="/root/autodl-tmp/conda-envs/emix-reinstall"
EVAL_ENV="/root/autodl-tmp/conda-envs/emix-twin-reinstall"
CONDA="/root/miniconda3/bin/conda"
def spec(): return yaml.safe_load(SPEC.read_text())
def prepare(s):
    tasks=s["sources"]["tasks"]; validate_sources(s["sources"]["dataset_root"],list(tasks)); GEN.mkdir(parents=True,exist_ok=True); t=s["training"]
    train={"batch_size":t["batch_size"],"num_workers":t["num_workers"],"steps":t["steps"],"save_freq":t["save_freq"],"log_freq":t["log_freq"],"output_dir":str(TRAIN),"policy":{"type":"pi05","push_to_hub":False,"pretrained_path":t["pretrained_path"],"device":t["device"],"optimizer_lr":t["optimizer_lr"],"n_obs_steps":t["n_obs_steps"],"chunk_size":t["chunk_size"],"n_action_steps":t["n_action_steps"],"dtype":t["dtype"],"gradient_checkpointing":t["gradient_checkpointing"],"scheduler_warmup_steps":t["scheduler_warmup_steps"],"scheduler_decay_steps":t["scheduler_decay_steps"],"scheduler_decay_lr":t["scheduler_decay_lr"]},"dataset":{"root":s["sources"]["dataset_root"],"repo_id":next(iter(tasks)),"repo_ids":list(tasks),"video_backend":t["video_backend"]},"protocol":{**s["protocol"],"arm":"pi05-lora"},"peft":s["peft"]}
    (GEN/"train.yaml").write_text(yaml.safe_dump(train,sort_keys=False))
    for step in s["milestones"]:
      for task,prompt in tasks.items():
       ev={"dataset":{"repo_id":task,"root":str(Path(s["sources"]["dataset_root"])/task)},"evaluation":{"checkpoint":str(TRAIN/"checkpoints"/f"{step:06d}"/"pretrained_model"),"output_dir":str(OUT/"eval"/f"{step:06d}"/task),"benchmark_root":str(ROOT/"third_party/RoboTwin"),"task_name":task,"task_config":task,"task_prompt":prompt,"piper_distance":s["evaluation"]["piper_distance"],"episodes":s["evaluation"]["episodes"],"seed":s["evaluation"]["seed"],"device":s["evaluation"]["device"],"save_video":s["evaluation"]["save_video"],"video_fps":s["evaluation"]["video_fps"]},"protocol":{**s["protocol"],"milestone_step":step,"task":task}}
       (GEN/f"eval_{step:06d}_{task}.yaml").write_text(yaml.safe_dump(ev,sort_keys=False))
    print(f"Prepared PI0.5 LoRA config and {len(s['milestones'])*len(tasks)} eval configs",flush=True)
def complete(out,ckpt,episodes):
    for m in out.glob("*/metrics.json") if out.exists() else []:
      try:
       x=json.loads(m.read_text())
       if x.get("episodes")==episodes and Path(x["checkpoint"]).resolve()==ckpt.resolve(): return True
      except Exception: pass
    return False
def run(s):
    from huggingface_hub import hf_hub_download
    # PI05Policy.from_pretrained in this LeRobot build can fall back to random
    # initialization after a Hub download error, so fail closed before training.
    hf_hub_download(repo_id=s["training"]["pretrained_path"], filename="model.safetensors")
    # PI05 uses the gated Google PaliGemma tokenizer; check access before GPU training.
    hf_hub_download(repo_id="google/paligemma-3b-pt-224", filename="config.json")
    OUT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env["HF_HOME"]="/root/autodl-tmp/hf-cache"; env["HF_TOKEN_PATH"]="/root/.cache/huggingface/token"; env["EMIX_POLICY_WORKER_ENV"]=ENV; ckroot=TRAIN/"checkpoints"
    for step in s["milestones"]:
      cp=ckroot/f"{step:06d}"; model=cp/"pretrained_model"
      if not (model/"adapter_model.safetensors").is_file() and not (model/"model.safetensors").is_file():
       args=[CONDA,"run","--no-capture-output","-p",ENV,"python","-u","train/train.py","--config",str(GEN/"train.yaml"),"--seed",str(s["protocol"]["training_seed"]),"--total_steps",str(step)]; last=ckroot/"last"
       if last.is_symlink(): args += ["--resume_from",str(last.resolve())]
       with (LOG/f"train_to_{step}.log").open("a") as f: subprocess.run(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
      assert (model/"config.json").is_file(), f"missing checkpoint config at {step}"
      for task in s["sources"]["tasks"]:
       dest=OUT/"eval"/f"{step:06d}"/task
       if complete(dest,model,s["evaluation"]["episodes"]): continue
       args=[CONDA,"run","--no-capture-output","-p",EVAL_ENV,"python","-u","evaluation/evaluate.py","--config",str(GEN/f"eval_{step:06d}_{task}.yaml")]
       with (LOG/f"eval_{step}_{task}.log").open("a") as f: subprocess.run(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
       if not complete(dest,model,s["evaluation"]["episodes"]): raise RuntimeError(f"No complete metrics: {step}/{task}")
    (OUT/".complete").write_text("PI0.5 LoRA checkpoints evaluated across ten tasks.\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=["prepare","mock","run"],required=True);a=p.parse_args();s=spec();prepare(s)
 if a.mode=="mock":
  from embodied_mixup.policies.train import _build_train_config
  c=_build_train_config(yaml.safe_load((GEN/"train.yaml").read_text()),GEN/"train.yaml"); assert c.policy_cfg.type=="pi05" and len(c.dataset_repo_ids)==10 and c.steps==40000 and c.peft["r"]==16
  from embodied_mixup.policies.utils.multi_dataset import load_multi_dataset
  meta,ds=load_multi_dataset(c.dataset_root,c.dataset_repo_ids,c.policy_cfg,c.video_backend); x=ds[0]; print("Config/data valid",len(ds),x["task"],tuple(x["action"].shape),flush=True)
 if a.mode=="run": run(s)
if __name__=="__main__": main()
