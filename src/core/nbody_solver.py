import logging

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

if device.type == "cpu":
    logger.warning(
        "CUDA not available. Using CPU instead. Install a CUDA-enabled PyTorch build for GPU acceleration."
    )

N = 5000
G = 1.0
dt = 0.01
softening = 1

pos = torch.randn(N, 2, device=device)
vel = torch.randn(N, 2, device=device) * 0.1
mass = torch.ones(N, device=device)


def compute_acceleration(pos):
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist_sq = (diff**2).sum(-1) + softening
    inv_dist3 = dist_sq.pow(-1.5)
    force = -G * diff * inv_dist3.unsqueeze(-1)
    acc = (force * mass.unsqueeze(0).unsqueeze(-1)).sum(1)
    return acc


fig, ax = plt.subplots()
scat = ax.scatter([], [])
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect("equal")


def update(frame):
    global pos, vel
    acc = compute_acceleration(pos)
    vel += acc * dt
    pos += vel * dt

    scat.set_offsets(pos.detach().cpu().numpy())
    return (scat,)


ani = FuncAnimation(fig, update, interval=10)
plt.show()
