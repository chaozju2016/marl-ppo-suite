import sys
import os
from datetime import datetime
import imageio
import numpy as np
import time
from collections import deque
import pandas as pd
import torch
from typing import Union


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

# Conditional import for wandb
try:
    import wandb
except ImportError:
    wandb = None # type: ignore[misc, assignment]

class Logger:
    """
    Logging class to monitor training, supporting TensorBoard and optional CSV logging.
    
    Args:
        run_name (str): Unique identifier for the run. Defaults to current timestamp.
        folder (str): Root directory for storing logs. Defaults to 'runs'.
        algo (str): Algorithm name, used in directory structure. Defaults to 'sac'.
        env (str): Environment name, used in directory structure. Defaults to 'Env'.
        save_csv (bool): Whether to log metrics to a CSV file. Defaults to False.
    """
    
    def __init__(
         self,
        run_name=datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        folder="runs",
        algo="sac",
        env="Env",
        save_csv=False,
        use_wandb=False,
        wandb_project="default",
        wandb_entity=None,
        config=None
    ):
        self.run_name = run_name
        self.dir_name = os.path.join(folder, env, algo, run_name)
        os.makedirs(self.dir_name, exist_ok=True)

        self.writer = SummaryWriter(self.dir_name)
        self.name_to_values = {}  # Stores deque of recent values for smoothing
        self.current_env_step = 0
        self.start_time = time.time()
        self.last_csv_save = time.time()
        self.save_csv = save_csv
        self.save_every = 10 * 60  # Save CSV every 10 seconds

        if self.save_csv:
            self._data = {}  # {step: {key: val, ...}, ...} for CSV logging
        
        self.use_wandb = use_wandb and (wandb is not None) # Ensure wandb is imported
        if self.use_wandb:
            self.wb = wandb.init(
                name=run_name, 
                project=wandb_project, 
                entity=wandb_entity,
                dir = self.dir_name, #keeps artifacts in the same folder
                config=config, #hyper-parameters in UI
                sync_tensorboard=True # one-line TB-sync
            )
            # Define metrics that use step
            self.wb.define_metric("global_step")
            # Define all common metrics to use global_step as x-axis
            self.wb.define_metric("*", step_metric="global_step")
        
        if config is not None:
            self.log_all_hyperparameters(config)

    def log_all_hyperparameters(self, hyperparams: dict):
        """Log hyperparameters to TensorBoard and print them to stdout."""
        self.add_hyperparams(hyperparams)
        self.log_hyperparameters(hyperparams)

    def add_hyperparams(self, hyperparams: dict):
        """Log hyperparameters to TensorBoard."""
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in hyperparams.items()])),
        )
    
    def log_hyperparameters(self, hyperparams: dict):
        """Pretty print hyperparameters in a table format."""
        hyper_param_space, value_space = 30, 40
        format_str = "| {:<" + f"{hyper_param_space}" + "} | {:<" + f"{value_space}" + "}|"
        hbar = "-" * (hyper_param_space + value_space + 6)
    
        print(hbar)
        print(format_str.format("Hyperparams", "Values"))
        print(hbar)
    
        for key, value in hyperparams.items():
            print(format_str.format(str(key), str(value)))
    
        print(hbar)

    def add_run_command(self):
        """Log the terminal command used to start the run."""
        cmd = " ".join(sys.argv)
        self.writer.add_text("terminal", cmd)
        with open(os.path.join(self.dir_name, "cmd.txt"), "w") as f:
            f.write(cmd)
    
    def log_training(self, data: dict, print_to_stdout: bool = False):
        """Log training metrics to TensorBoard and optionally to CSV."""
        for key, val in data.items():
            self.add_scalar(key, val, self.current_env_step)
        if print_to_stdout: self.log_stdout()

    def add_scalar(self, key: str, val: float, step: int, smoothing: bool = True):
        """
        Log a scalar value to TensorBoard and optionally to CSV.
        
        Args:
            key (str): Metric name (e.g., 'loss')
            val (float): Value to log
            step (int): Current training step
            smoothing (bool): Whether to smooth values for stdout logging. Defaults to True.
        """
        # Log to TensorBoard
        self.writer.add_scalar(key, val, step)
        if self.use_wandb:
            self.wb.log({
                "global_step": step,
                key: val
                })

        # Update smoothing deque
        if key not in self.name_to_values:
            self.name_to_values[key] = deque(maxlen=5 if smoothing else 1)
        self.name_to_values[key].append(val)
        self.current_env_step = max(self.current_env_step, step)

        # Log to CSV if enabled
        if self.save_csv:
            if step not in self._data:
                self._data[step] = {}
            self._data[step][key] = val  # Store raw value
            
            # Periodically save CSV
            if time.time() - self.last_csv_save > self.save_every:
                self.save2csv()
                self.last_csv_save = time.time()
    
    def add_video(self, tag: str, frames: Union[np.ndarray, torch.Tensor],
                  step: int, fps: int = 30):
        """
        Log video clips to TensorBoard and W&B.
        - frames: np.ndarray or torch.Tensor of shape
            * (T, H, W)         grayscale single clip
            * (T, H, W, C)      C=1/3/4 single clip
            * (T, C, H, W)      single clip
            * (N, T, H, W, C)   batch of clips
        """
        # 1) Bring to numpy uint8 (N, T, H, W, C)
        if isinstance(frames, torch.Tensor):
            arr = frames.cpu().numpy()
        else:
            arr = np.asarray(frames)

        # Cast floats → [0,255] uint8
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)

        # Now handle dims
        if arr.ndim == 3:  # (T, H, W) → grayscale
            T, H, W = arr.shape
            arr = arr.reshape(T, H, W, 1)

        if arr.ndim == 4:
            # could be (T, H, W, C) or (T, C, H, W)
            T, D1, D2, D3 = arr.shape
            if D3 in (1, 3, 4):
                # (T, H, W, C) → add batch
                arr = arr[None]  # → (1, T, H, W, C)
            elif D1 in (1, 3, 4):
                # (T, C, H, W) → reorder then batch
                arr = np.transpose(arr, (0, 2, 3, 1))[None]  # → (1, T, H, W, C)
            else:
                raise ValueError(f"{tag}: can't infer channels from shape {arr.shape}")

        if arr.ndim == 5:
            # assume (N, T, H, W, C)
            N, T, H, W, C = arr.shape
            if C not in (1, 3, 4):
                raise ValueError(f"{tag}: last dim={C} is not a valid channel count")
        else:
            raise ValueError(f"{tag}: expected 3–5 dims after preprocess, got {arr.shape}")

        # 2) TensorBoard needs (N, T, C, H, W)
        tb = torch.from_numpy(arr).permute(0, 1, 4, 2, 3)
        self.writer.add_video(tag, tb, global_step=step, fps=fps)

        # 3) W&B: write MP4 and log
        if self.use_wandb:
            # pick first clip
            clip = arr[0]  # (T, H, W, C)
            out = os.path.join(self.dir_name, f"{tag.replace('/','_')}_{step}.mp4")
            # enforce macro_block_size=1 to avoid the 16×16 resize warning
            with imageio.get_writer(out, 
                                    fps=fps, 
                                    codec="libx264", 
                                    macro_block_size=1) as w:
                for frame in clip:
                    w.append_data(frame)
            video_obj = wandb.Video(out, format="mp4")
            self.wb.log({tag: video_obj})

    def save2csv(self, file_name: str = None):
        """Save logged data to a CSV file."""
        if not self.save_csv or not self._data:
            return
        
        if file_name is None:
            file_name = os.path.join(self.dir_name, "progress.csv")
        
        # Convert to DataFrame
        steps = sorted(self._data.keys())
        rows = []
        for step in steps:
            row = {'global_step': step}
            row.update(self._data[step])
            rows.append(row)
        df = pd.DataFrame(rows)
        
        # Ensure 'global_step' is first column
        cols = ['global_step'] + [c for c in df.columns if c != 'global_step']
        df = df[cols]
        
        df.to_csv(file_name, index=False)
    
    def close(self):
        """Close the TensorBoard writer and save CSV."""
        self.writer.close()
        if self.use_wandb:
            self.wb.finish()
        if self.save_csv:
            self.save2csv()
    
    def log_stdout(self):
        """Print smoothed metrics to stdout."""
        results = {k: np.mean(v) for k, v in self.name_to_values.items()}
        results['step'] = self.current_env_step
        # results['fps'] = self.fps()
        pprint(results)
    
    def fps(self) -> int:
        """Calculate frames per second (steps per second)."""
        elapsed = time.time() - self.start_time
        return int(self.current_env_step / elapsed) if elapsed > 0 else 0


def pprint(dict_data):
    """Pretty print metrics in a table format."""
    key_space, val_space = 40, 40
    border = "-" * (key_space + val_space + 5)
    row_fmt = f"| {{:<{key_space}}} | {{:<{val_space}}}|"
    
    print(f"\n{border}")
    for k, v in dict_data.items():
        k_str = truncate_str(str(k), key_space)
        v_str = truncate_str(str(v), val_space)
        print(row_fmt.format(k_str, v_str))
    print(f"{border}\n")


def truncate_str(s: str, max_len: int) -> str:
    """Truncate string with ellipsis if exceeds max length."""
    return s if len(s) <= max_len else s[:max_len-3] + "..."