"""
Export the best-model artefact + media from W&B to Hugging Face Hub
and create a CleanRL-style model card.
"""
import argparse, shutil, json, textwrap, os, re
from pathlib import Path
import dateutil
import numpy as np
import tqdm
import wandb
from huggingface_hub import HfApi, Repository, upload_folder
from jinja2 import Template
from dotenv import load_dotenv

CARD = Template("""\
---
license: mit
library_name: marl-ppo-suite
tags:
- reinforcement-learning
- starcraft-mac
- {{ env_name }}
- {{ algo|lower }}
- {{ map_name }}
- {{ scenario }}
model-index:
- name: {{ algo }} on {{ scenario }}
  results:
  - task:
      type: reinforcement-learning
      name: StarCraft Multi-Agent Challenge v2
    dataset:
      name: {{ map_name }}
      type: smacv2
    metrics:
      - name: win-rate
        type: win_rate
        value: {{ win_rate|round(3) }}
      - name: mean-reward
        type: mean_reward
        value: {{ reward|round(2) }}
      - name: mean-ep-length
        type: mean_episode_length
        value: {{ length|round(1) }}
---

# {{ algo }} on **{{ scenario }}**

*10 M environment steps · {{ wall|round(2) }} h wall-clock · seed {{ seed }}*

This is a trained model of a `{{ algo }}` agent playing *{{ scenario }}*.  
The model was produced with the open-source
[`marl-ppo-suite`](https://github.com/legalaspro/marl-ppo-suite) training
code.

{% if wandb_public %}
[![Weights & Biases](https://img.shields.io/badge/%F0%9F%AA%9C Run-orange?logo=weightsandbiases)]({{ wandb_url }})
{% endif %}

## Usage  – quick evaluation / replay

```bash
# 1. install the codebase (directly from GitHub)
pip install "marl-ppo-suite @ git+https://github.com/legalaspro/marl-ppo-suite"

# 2. get the weights & config from HF
wget https://huggingface.co/<repo-id>/resolve/main/final-torch.model
wget https://huggingface.co/<repo-id>/resolve/main/config.json

# 3-a. Generate a StarCraft II replay file 1 episode in starcraft replay folder
marl-train \
  --mode render \
  --model final-torch.model \
  --config config.json \
  --render_episodes 1 \       

# 3-b. generate additionally video drawn from frames 
marl-train \
  --mode render \
  --model final-torch.model \
  --config config.json \
  --render_episodes 1 \
  --render_mode rgb_array       
```

## Files

* **`final-torch.model`** – PyTorch checkpoint  
{% if video_name %}* **`{{ video_name }}`** – gameplay of the final policy{% endif %} 
* **`config.json`** – training config
* **`tensorboard/`** – full logs

## Hyper-parameters

```python
{{ hps | tojson(indent=2) }}
```
                

""")

# ───────────────────────── CLI ──────────────────────────
def cli() -> argparse.Namespace:
    load_dotenv(Path(".env"))          # silent‑no‑file

    p = argparse.ArgumentParser()
    p.add_argument("--run_id",  required=True,
                   help="W&B run id (e.g. n2x9y78f)")
    p.add_argument("--entity",  default=os.getenv("WANDB_ENTITY"),
                   help="W&B entity; env WANDB_ENTITY if unset")
    p.add_argument("--project", default=os.getenv("WANDB_PROJECT","marl-ppo-suite"),
                   help="W&B project; env WANDB_PROJECT if unset")
    p.add_argument("--api_key", default=os.getenv("WANDB_API_KEY"),
                   help="Override WANDB_API_KEY")
    p.add_argument("--hf_token", default=os.getenv("HF_API_TOKEN"),
                   help="Override Hugging Face token")
    p.add_argument("--hf_org", default="legalaspro",
                   help="Hugging Face organization")

    return p.parse_args()

# ───────────────────────── helpers ──────────────────────
def wandb_login(api_key: str|None):
    if api_key: os.environ["WANDB_API_KEY"] = api_key
    if "WANDB_API_KEY" not in os.environ:
        raise SystemExit("❌ Need WANDB_API_KEY (flag/env/.env)")
    wandb.login()

def rebuild_tb(run, tb_dir):
    """
    Re-create a TensorBoard log directory from an *already-finished*
    Weights-and-Biases run.
    wandb sync <path-to-events-file> --project marl-ppo-suite --entity legalaspro-rl


    Parameters
    ----------
    run     : wandb.apis.public.Run
    tb_dir  : Path | str          where .tfevents will be written
    """
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(tb_dir)
    chunk = 1000
    

    print("Downloading history …")
    if isinstance(run.created_at, str):
        # Parse the ISO format string to datetime
        created_at_timestamp = dateutil.parser.parse(run.created_at).timestamp()
    else:
        # Already a datetime object
        created_at_timestamp = run.created_at.timestamp()

    df = run.history(keys=[], pandas=True)

    steps     = df["_step"].to_numpy()
    wall_time = created_at_timestamp + df["_runtime"].to_numpy()

    scalar_cols = [c for c in df.columns
                   if not c.startswith("_")
                   and np.issubdtype(df[c].dtype, np.number)]

    for tag in tqdm.tqdm(scalar_cols, desc="writing .tfevents"):
        values = df[tag].to_numpy()

        # drop NaNs to keep the file small
        mask   = ~np.isnan(values)
        if not mask.any():
            continue

        for i in range(0, mask.sum(), chunk):
            sl = slice(i, i+chunk)
            for v, s, t in zip(values[mask][sl], steps[mask][sl], wall_time[mask][sl]):
                writer.add_scalar(tag, float(v), global_step=int(s), walltime=float(t))

    writer.flush()
    print(f"✅ wrote TensorBoard logs → {tb_dir}\n   open with: tensorboard --logdir {tb_dir}")



# ───────────────────────── main logic ───────────────────
def main():
    args = cli()
    wandb_login(api_key=args.api_key)                       # uses WANDB_API_KEY
    
    api   = wandb.Api()
    run   = api.run(f"{args.entity}/{args.project}/{args.run_id}")   # e.g. "legalaspro-rl/marl-ppo-suite/abc123"
    
    cfg, summary = run.config, run.summary

    algo      = cfg.get("algo", "MAPPO").upper()
    env_name  = cfg.get("env_name", "smacv2")
    map_name  = cfg.get("map_name", "unknown")
    seed      = cfg.get("seed", 0)
    win_rate  = summary.get("summary/eval/win_rate", 0.0)
    reward    = summary.get("summary/eval/rewards", 0.0)
    length    = summary.get("summary/eval/length", 0.0)
    wall      = summary.get("wall_clock_hours", 0.0)

    scenario = f"{env_name}_{map_name}"
    # 1.  gather artefacts --------------------------------------------------
    # tmp = Path(tempfile.mkdtemp())
    work = Path(".hf_tmp") / run.id          # stays inside your project
    work.mkdir(parents=True, exist_ok=True)

    # ---- config -----------------------------------------------------------
    (work / "config.json").write_text(json.dumps(cfg, indent=2))

    # ---- final model --------------------------------------------------------
    final = next(a for a in run.logged_artifacts() if "final" in a.name)
    final.download(root=work)

    # ---- pick the newest eval/render video --------------------------------
    mp4_files = sorted(
        [f for f in run.files() if re.match(r"media/videos/eval.+\.mp4", f.name)],
        key=lambda f: f.updated_at,          # newest last
    )
    if mp4_files:
        # Target path for the video
        video_path = work / "replay.mp4"
        video_file = mp4_files[-1].download(root=work, replace=True)
        shutil.move(Path(video_file.name), video_path)
        shutil.rmtree(work / "media", ignore_errors=True)
                
        video_name = "replay.mp4"
    else:
        video_name = ""     # card will omit the <video> tag

    # ---------- TensorBoard ----------
    tb_dir = work / "tensorboard"
    tb_dir.mkdir(exist_ok=True)

    downloaded_tensorboard = False
    for f in run.files():
        if f.name.startswith("events.out.tfevents"):
            f.download(root=tb_dir, replace=True)
            downloaded_tensorboard = True
    
    if not downloaded_tensorboard:   # fallback
        print("No .tfevents in W&B – Rebuilding from W&B history …")
        rebuild_tb(run, tb_dir)

    # 2.  write model card --------------------------------------------------
    hp_keys = [
        "lr", "gamma", "gae_lambda", "clip_param", "entropy_coef", 
        "hidden_size", "fc_layers", "n_steps", "ppo_epoch", "num_mini_batch", "seed",
        "use_rnn", "data_chunk_length", "state_type", "use_value_norm", 
        "use_reward_norm", "reward_norm_type", "value_norm_type"
    ]
 
    card = CARD.render(
        algo=algo, map_name=map_name, env_name=env_name,
        wall=wall, seed=seed, scenario=scenario,
        win_rate=win_rate, reward=reward, length=length,
        wandb_url=run.url,
        video_name = video_name if video_name else "_(no-video)_",
        hps = {k: cfg[k] for k in hp_keys if k in cfg},
        wandb_public = False,
    )
    (work / "README.md").write_text(textwrap.dedent(card))
    # 3.  push to HF --------------------------------------------------------
    repo_id = f"{args.hf_org}/{algo}-{scenario}-seed{seed}"
    api = HfApi(token=args.hf_token)
    

    if not api.repo_exists(repo_id):
        api.create_repo(repo_id, exist_ok=True, repo_type="model")

    # repo = Repository(local_dir=work, clone_from=repo_id, token=args.hf_token)
    # repo.git_add(all=True)
    # repo.git_commit("Add model card, weights, video, TB")
    # repo.git_push()
    upload_folder(
        repo_id        = repo_id,
        folder_path    = work,
        path_in_repo   = ".",       # keep same structure
        token          = args.hf_token,
        commit_message = "Add model card, weights, video, TB",
    )
    print("✅ pushed to", f"https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()