"""
Unified training utilities: logging, plotting, and checkpointing.

Example Usage:
---
from train_utils import Tracker

t = Tracker("my_experiment")  # or auto-generates timestamp

# Register model/optimizer for checkpointing
t.register("model", model)
t.register("optimizer", optimizer)

for epoch in range(epochs):
    # Log metrics (single or batched)
    t.log(epoch, train_loss=avg_loss, lr=scheduler.get_last_lr()[0])
    t.log(epoch, val_loss=val_loss)

    # Save checkpoint (keeps last N, tracks best)
    t.save(epoch=epoch, is_best=(val_loss < best_loss), keep_last=3)

# Plotting
t.plot("train_loss")  # single metric
t.plot(["train_loss", "val_loss"], fname="losses", smooth=0.9)  # with EMA

# Save logs to disk
t.save_logs()  # saves both txt and json

# Loading a previous run
t.load("old_experiment")  # loads into registered objects
# or just get the checkpoint dict
ckpt = t.load("old_experiment", into_registry=False)
"""

import glob
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch


class Tracker:
    def __init__(
        self,
        run_name: str | None = None,
        log_dir: str = "logs",
        ckpt_dir: str = "checkpoints",
    ):
        self.run_name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(log_dir, self.run_name)
        self.ckpt_dir = os.path.join(ckpt_dir, self.run_name)

        self.data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.registry: dict[str, object] = {}

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def log(self, step: int, **kwargs) -> None:
        """Log one or more metrics at a given step."""
        for name, value in kwargs.items():
            self._add_stat(name, step, value)

    def _add_stat(self, name: str, x, y) -> None:
        if name in self.data:
            xs, ys = self.data[name]
            self.data[name] = (np.append(xs, x), np.append(ys, y))
        else:
            self.data[name] = (np.array([x]), np.array([y]))

    def save_logs(self, metrics: list[str] | None = None) -> None:
        """Save logged metrics to txt and json files."""
        os.makedirs(self.log_dir, exist_ok=True)
        metrics = metrics or list(self.data.keys())

        for name in metrics:
            if name not in self.data:
                print(f"[Tracker] warning: metric '{name}' not found")
                continue

            xs, ys = self.data[name]

            # txt format: "step value" per line
            txt_path = os.path.join(self.log_dir, f"{name}.txt")
            with open(txt_path, "w") as f:
                for x, y in zip(xs, ys):
                    f.write(f"{x} {y}\n")

            # json format: {"steps": [...], "values": [...]}
            json_path = os.path.join(self.log_dir, f"{name}.json")
            with open(json_path, "w") as f:
                json.dump({"steps": xs.tolist(), "values": ys.tolist()}, f)

    def load_logs(self, run_name: str | None = None) -> None:
        """Load logs from a previous run into self.data."""
        log_dir = os.path.join(os.path.dirname(self.log_dir), run_name or self.run_name)

        for json_file in glob.glob(os.path.join(log_dir, "*.json")):
            name = os.path.splitext(os.path.basename(json_file))[0]
            with open(json_file) as f:
                d = json.load(f)
                self.data[name] = (np.array(d["steps"]), np.array(d["values"]))

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot(
        self,
        metrics: str | list[str],
        fname: str | None = None,
        smooth: bool = False,
    ) -> None:
        """
        Plot one or more metrics.

        Args:
            metrics: metric name or list of names to plot
            fname: output filename (without extension)
            smooth: Smooth the graph
        """
        os.makedirs(self.log_dir, exist_ok=True)

        if isinstance(metrics, str):
            metrics = [metrics]
        fname = fname or "_".join(metrics)

        plt.figure(figsize=(10, 6), dpi=100)

        for name in metrics:
            if name not in self.data:
                print(f"[Tracker] warning: metric '{name}' not found")
                continue

            xs, ys = self.data[name]

            if smooth:
                ys_smooth = self._ema(ys, 0.5)
                plt.plot(xs, ys, alpha=0.3, color="gray")
                plt.plot(xs, ys_smooth, label=f"{name}")
                # Find min/max in smoothed data
                y_plot = ys_smooth
            else:
                plt.plot(xs, ys, label=name)
                y_plot = ys

            # Find and mark minimum and maximum points
            min_idx = np.argmin(y_plot)
            max_idx = np.argmax(y_plot)

            min_x, min_y = xs[min_idx], y_plot[min_idx]
            max_x, max_y = xs[max_idx], y_plot[max_idx]

            # Mark minimum point
            plt.plot(
                min_x,
                min_y,
                "o",
                color="red",
                markersize=8,
                label=f"{name} min" if len(metrics) == 1 else None,
            )
            plt.annotate(
                f"Min: {min_y:.4f}",
                xy=(min_x, min_y),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

            # Mark maximum point
            plt.plot(
                max_x,
                max_y,
                "s",
                color="blue",
                markersize=8,
                label=f"{name} max" if len(metrics) == 1 else None,
            )
            plt.annotate(
                f"Max: {max_y:.4f}",
                xy=(max_x, max_y),
                xytext=(10, -20),
                textcoords="offset points",
                fontsize=9,
                color="blue",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        plt.xlabel("step")
        plt.legend(loc="best")
        plt.tight_layout()

        plt.savefig(os.path.join(self.log_dir, f"{fname}.png"))
        plt.close()

    @staticmethod
    def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
        """Exponential moving average. Higher alpha = more smoothing."""
        result = np.zeros_like(values)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * result[i - 1] + (1 - alpha) * values[i]
        return result

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def register(self, name: str, obj) -> None:
        """Register an object (model, optimizer, scheduler) for checkpointing."""
        self.registry[name] = obj

    def save(
        self,
        is_best: bool = False,
        keep_last: int = 0,
        **metadata,
    ) -> str:
        """
        Save checkpoint with registered objects + metadata.

        Args:
            is_best: if True, also save as best.pt
            keep_last: if > 0, keep only the N most recent epoch_*.pt files
            **metadata: extra data to save (epoch, step, etc.)

        Returns:
            path to saved checkpoint
        """
        os.makedirs(self.ckpt_dir, exist_ok=True)

        payload = dict(metadata)
        for name, obj in self.registry.items():
            payload[name] = obj.state_dict()

        # Save as epoch_XXX.pt if epoch provided, else last.pt
        if "epoch" in metadata:
            ckpt_name = f"epoch_{metadata['epoch']:04d}.pt"
        else:
            ckpt_name = "last.pt"

        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        torch.save(payload, ckpt_path)

        if is_best:
            torch.save(payload, os.path.join(self.ckpt_dir, "best.pt"))

        if keep_last > 0:
            self._cleanup_old_checkpoints(keep_last)

        return ckpt_path

    def _cleanup_old_checkpoints(self, keep: int) -> None:
        pattern = os.path.join(self.ckpt_dir, "epoch_*.pt")
        ckpts = sorted(glob.glob(pattern))

        for old_ckpt in ckpts[:-keep]:
            os.remove(old_ckpt)

    def load(
        self,
        run_name: str | None = None,
        which: str = "best",
        into_registry: bool = True,
        map_location: str | torch.device | None = None,
    ) -> dict:
        """
        Load a checkpoint.

        Args:
            run_name: name of run to load (defaults to current run)
            which: "best", "last", or "epoch_0005" etc.
            into_registry: if True, load state_dicts into registered objects
            map_location: device to load tensors onto

        Returns:
            checkpoint dict with metadata and state_dicts
        """
        run_name = run_name or self.run_name
        ckpt_dir = os.path.join(os.path.dirname(self.ckpt_dir), run_name)

        ckpt_path = os.path.join(ckpt_dir, f"{which}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

        if into_registry:
            for name, obj in self.registry.items():
                if name in ckpt:
                    obj.load_state_dict(ckpt[name])
                else:
                    print(f"[Tracker] warning: '{name}' not found in checkpoint")

        return ckpt

    def list_checkpoints(self, run_name: str | None = None) -> list[str]:
        """List available checkpoints for a run."""
        run_name = run_name or self.run_name
        ckpt_dir = os.path.join(os.path.dirname(self.ckpt_dir), run_name)
        return [os.path.basename(p) for p in glob.glob(os.path.join(ckpt_dir, "*.pt"))]
