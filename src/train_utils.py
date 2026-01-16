"""
Example Usage:
---
from train_utils import Checkpointer, Tracker

run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

t = Tracker(run_name=run_name)
c = Checkpointer(run_name=run_name)

t.add_stat("train_loss", epoch, avg_loss)
t.save_txt("train_loss")
t.save_plot(["train_loss", "val_loss"], fname="loss")


c.register("model", model)
c.register("optimizer", optim)
c.save(dict(epoch=3), is_best=True)

meta = c.load("exp_001")
model.load_state_dict(meta["model"])
optim.load_state_dict(meta["optimizer"])
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch


class Checkpointer:
    def __init__(self, out_dir: str = "checkpoints", run_name: str | None = None):
        self.run_name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = out_dir
        self.run_dir = os.path.join(out_dir, self.run_name)
        self.registry: dict = {}

    def register(self, name: str, obj) -> None:
        self.registry[name] = obj

    def save(self, payload: dict, is_best=False):
        os.makedirs(self.run_dir, exist_ok=True)
        for name, obj in self.registry.items():
            payload[name] = obj.state_dict()

        torch.save(payload, os.path.join(self.run_dir, "last.pt"))

        if is_best:
            torch.save(payload, os.path.join(self.run_dir, "best.pt"))

    def load(self, run_name: str):
        run_dir = os.path.join(self.out_dir, run_name)
        best_path = os.path.join(run_dir, "best.pt")

        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Run not found: {run_name}")
        elif not os.path.exists(best_path):
            raise FileNotFoundError(f"Best not found: {run_name}")

        ckpt = torch.load(best_path)
        return ckpt


class Tracker:
    data: dict[str, tuple[np.ndarray, np.ndarray]]

    def __init__(self, out_dir="logs", run_name: str | None = None):
        self.run_name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = os.path.join(out_dir, self.run_name)
        self.data = {}

    def add_stat(self, name: str, x, y):
        if name in self.data:
            data = self.data[name]
            self.data[name] = (np.append(data[0], x), np.append(data[1], y))
        else:
            self.data[name] = (np.array([x]), np.array([y]))

    def save_txt(self, name: str, suffix: str | None = None):
        os.makedirs(self.out_dir, exist_ok=True)
        if name not in self.data:
            print(f"[StatsTracker] warning: Stat {name} not found")
            return
        suffix = f"_{suffix}" if suffix else ""

        with open(os.path.join(self.out_dir, f"{name}{suffix}.txt"), "w") as f:
            for x, y in zip(self.data[name][0], self.data[name][1]):
                f.write(f"{x} {y}\n")

    def save_plot(self, name: str | list[str], fname: str | None = None):
        os.makedirs(self.out_dir, exist_ok=True)
        if isinstance(name, str):
            name = [name]
        fname = fname or "_".join(name)

        plt.figure(figsize=(8, 6), dpi=80)
        for n in name:
            if n not in self.data:
                print(f"[StatsTracker] warning: Stat {n} not found")
                continue
            x, y = self.data[n]
            plt.plot(x, y, label=n)

        plt.legend(loc="upper left")
        plt.savefig(os.path.join(self.out_dir, f"{fname}.png"))
        plt.close()
