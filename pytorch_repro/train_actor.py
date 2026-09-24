import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import ActorNet


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_state(x: np.ndarray) -> np.ndarray:
    """Map legacy x_pretrain [util, remaining_time] -> [price, util, remaining_time].

    The old TF script appended a zero channel, yielding [util, rt, 0].
    For the modern reproduction we put the zero price in the semantically
    correct first column.
    """
    price = np.zeros((*x.shape[:-1], 1), dtype=np.float32)
    return np.concatenate([price, x.astype(np.float32)], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../RL-DDPG/data")
    ap.add_argument("--output", default="actor_pretrained.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    seed_all(args.seed)
    data_dir = Path(args.data_dir)
    x = np.load(data_dir / "x_pretrain.npy")
    y = np.load(data_dir / "y_pretrain.npy").astype(np.float32)
    state = make_state(x)

    # The production actor in chasing.py used action_range=40.
    y = np.clip(y, -40.0, 40.0)

    idx = np.random.permutation(len(state))
    split = int(0.8 * len(idx))
    tr, va = idx[:split], idx[split:]

    train_ds = TensorDataset(
        torch.from_numpy(state[tr]),
        torch.from_numpy(y[tr, None]),
    )
    val_x = torch.from_numpy(state[va])
    val_y = torch.from_numpy(y[va, None])

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = ActorNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    best = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
            count += len(xb)

        model.eval()
        with torch.no_grad():
            pred = model(val_x.to(device))
            val_mse = float(mse(pred, val_y.to(device)))
            denom = torch.clamp(val_y.to(device).abs(), min=1.0)
            val_mape = float(((pred - val_y.to(device)).abs() / denom).mean())

        if val_mse < best:
            best = val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train_mse": total / max(count, 1),
                        "val_mse": val_mse,
                        "val_mape": val_mape,
                        "device": device,
                    },
                    sort_keys=True,
                )
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "state_dim": 3,
            "hidden_size": 128,
            "num_layers": 2,
            "c_dim": 128,
            "action_range": 40.0,
            "seed": args.seed,
            "best_val_mse": best,
        },
        out,
    )
    print(f"saved={out} best_val_mse={best:.6f}")


if __name__ == "__main__":
    main()
