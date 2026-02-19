import logging
import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from matplotlib.animation import FuncAnimation
from PIL import Image
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M%S",
)
logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Cihaz: {device}")


@dataclass
class NBody:
    """
    A class for simulating N-Body Problems using PyTorch GPU acceleration.

    Attributes:
    N: Particle amount
    G: Gravitational force
    dt: Delta time
    softening:
    """

    N: int
    G: float
    dt: float
    softening: float

    # Pre-allocated tensors for performance
    pos: torch.Tensor = field(init=False)
    vel: torch.Tensor = field(init=False)
    mass: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.vel = torch.randn((self.N, 2), device=device) * 0.05

    @staticmethod
    def get_initial_conditions(img_name, num_particles):
        base_path = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base_path, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"'{img_name}' is unavailable! path: {img_path}")

        img = Image.open(img_path).convert("L")
        torch.nn.AvgPool2d(kernel_size=10, stride=2)(transforms.ToTensor()(img))

        img_tensor = transforms.ToTensor()(img).to(device).squeeze()

        weights = 1.0 - img_tensor
        weights = weights.pow(3)

        prob = weights.view(-1) / weights.sum()

        indices = torch.multinomial(prob, num_particles, replacement=True)

        y = (indices // weights.shape[1]).float()
        x = (indices % weights.shape[1]).float()

        pos = torch.stack([x, y], dim=1)
        pos = (pos / weights.shape[0]) * 10 - 5

        m = weights.view(-1)[indices]

        return pos, m

    def compute_acceleration(self, pos, mass):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist_sq = (diff**2).sum(-1) + self.softening
        inv_dist3 = dist_sq.pow(-1.5)

        force = -self.G * diff * inv_dist3.unsqueeze(-1)
        acc = (force * mass.unsqueeze(0).unsqueeze(-1)).sum(1)
        return acc

    def update(self, frame):
        acc = self.compute_acceleration(self.pos, self.mass)
        self.vel += acc * self.dt
        self.pos += self.vel * self.dt

        scat.set_offsets(self.pos.detach().cpu().numpy())
        return (scat,)


if __name__ == "__main__":
    engine = NBody(N=10000, G=0.5, dt=0.02, softening=0.01)
    try:
        engine.pos, engine.mass = engine.get_initial_conditions("input.png", engine.N)
    except Exception as e:
        logger.error(e)
        exit()

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    scat = ax.scatter([], [], s=1, c="white", alpha=0.8)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.axis("off")

    ani = FuncAnimation(fig, engine.update, interval=1, blit=True)
    plt.show()
