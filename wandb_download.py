#!/usr/bin/env python
"""
Download model artefacts + config from a W&B run, organised as:

  ./artifacts/<algo>_<env>_<map>_<runID>/
        ├── best-torch.model   (if present)
        ├── final-torch.model  (if present and different)
        └── config.json
"""

from __future__ import annotations
import os, argparse, json
from pathlib import Path
import wandb
from dotenv import load_dotenv

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
    p.add_argument("--out",     default="artifacts",
                   help="Target directory")
    return p.parse_args()

# ───────────────────────── helpers ──────────────────────
def wandb_login(api_key: str|None):
    if api_key: os.environ["WANDB_API_KEY"] = api_key
    if "WANDB_API_KEY" not in os.environ:
        raise SystemExit("❌ Need WANDB_API_KEY (flag/env/.env)")
    wandb.login()


def pick_model_artifacts(run) -> dict[str,wandb.Artifact]:
    """
    Return {"best": Artifact|None, "final": Artifact|None}
      – if no explicit best/final names, first 'model' artefact becomes 'best'
    """
    out = {"best": None, "final": None}
    for art in run.logged_artifacts():
        if art.type != "model": continue
        name = art.name.lower()
        if "best"  in name and out["best"]  is None: out["best"]  = art
        if "final" in name and out["final"] is None: out["final"] = art
    # fallback
    if out["best"] is None and out["final"] is None:
        for art in run.logged_artifacts():
            if art.type == "model":
                out["best"] = art; break
    return out

# ───────────────────────── main logic ───────────────────
def main() -> None:
    args = cli()
    wandb_login(args.api_key)

    api  = wandb.Api()
    run  = api.run(f"{args.entity}/{args.project}/{args.run_id}")

    cfg  = run.config
    algo = cfg.get("algo",'unknown')
    env  = cfg.get("env_name",'unknown')
    mp   = cfg.get("map_name",'unknown')

    dest = Path(args.out, f"{algo}_{env}_{mp}_{run.id}")
    dest.mkdir(parents=True, exist_ok=True)

    artifacts = pick_model_artifacts(run)

    for key, art in artifacts.items():
        if art is None: continue
        art.download(root=dest)

    (dest / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"✅ downloaded to {dest}")

    print(f"\nrun  : {run.url}")
    print(f"model: {next(dest.glob('*torch.*'), '— none —')}")
    print("To render:")
    print(f"python train.py --mode render --config {dest/'config.json'} "
          f"--model {dest/'best-torch.model'}")
    print("To evaluate:")
    print(f"python train.py --mode eval --config {dest/'config.json'} "
          f"--model {dest/'best-torch.model'}")

# ───────────────────────── entry ────────────────────────
if __name__ == "__main__":
    main()