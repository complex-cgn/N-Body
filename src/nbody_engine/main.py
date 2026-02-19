import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import logging

# 📝 Log ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🚀 Cihaz seçimi (GPU varsa uçar, yoksa CPU ile devam)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Cihaz: {device}")

# --- ⚙️ PARAMETRELER ---
N = 5000          # Partikül sayısı (GPU gücüne göre artırabilirsin)
G = 0.5           # Yerçekimi sabiti
dt = 0.02         # Zaman adımı
softening = 0.01   # Sonsuz çekimi engellemek için yumuşatma

# --- 🖼️ RESİM İŞLEME VE BAŞLANGIÇ KOŞULLARI ---
def get_initial_conditions(img_name, num_particles):
    # 📍 Dinamik yol bulma (Klasör hatasını önler)
    base_path = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_path, img_name)
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ '{img_name}' bulunamadı! Yol: {img_path}")

    # Resmi gri tonlamalı aç ve boyutlandır
    img = Image.open(img_path).convert("L")
    img = img.resize((200, 200)) 
    
    img_tensor = T.ToTensor()(img).to(device).squeeze()
    
    # 🌑 Koyu pikselleri daha ağır yap (Ters çevir)
    weights = 1.0 - img_tensor
    weights = weights.pow(3)  # Kontrastı artırarak hatları belirginleştir
    
    # Olasılık dağılımı
    prob = weights.view(-1) / weights.sum()
    
    # 🎲 Ağırlığa göre rastgele pozisyon seçimi
    indices = torch.multinomial(prob, num_particles, replacement=True)
    
    y = (indices // weights.shape[1]).float()
    x = (indices % weights.shape[1]).float()
    
    # Koordinatları -5 ile 5 arasına normalize et
    pos = torch.stack([x, y], dim=1)
    pos = (pos / weights.shape[0]) * 10 - 5
    
    # Kütleleri ata (Koyu yerdeki partiküller daha ağır)
    m = weights.view(-1)[indices]
    
    return pos, m

# 🛠️ Başlangıç Verilerini Yükle
try:
    pos, mass = get_initial_conditions("input.png", N)
    vel = torch.randn((N, 2), device=device) * 0.05 # Çok hafif bir ilk hareket
except Exception as e:
    logger.error(e)
    exit()

# --- 🌌 FİZİK MOTORU ---
def compute_acceleration(pos, mass):
    # N-Body Vektörizasyonu (Pytorch Magic)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist_sq = (diff**2).sum(-1) + softening
    inv_dist3 = dist_sq.pow(-1.5)
    
    # F = G * (m1*m2) / r^2
    force = -G * diff * inv_dist3.unsqueeze(-1)
    acc = (force * mass.unsqueeze(0).unsqueeze(-1)).sum(1)
    return acc

# --- 🎨 GÖRSELLEŞTİRME ---
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
scat = ax.scatter([], [], s=1, c='white', alpha=0.8)

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.axis('off') # Eksenleri gizle, uzay hissi versin

def update(frame):
    global pos, vel
    # Fizik adımları
    acc = compute_acceleration(pos, mass)
    vel += acc * dt
    pos += vel * dt

    # Veriyi CPU'ya gönderip çizdir
    scat.set_offsets(pos.detach().cpu().numpy())
    return (scat,)

# 🎞️ Animasyonu Başlat
ani = FuncAnimation(fig, update, interval=1, blit=True)
plt.show()