from diffusers import DDPMScheduler, UNet2DModel
import torch
from torchvision.utils import save_image
import os

# 加载模型

image_size = 256
device = "cuda:5"
model = UNet2DModel.from_pretrained("checkpoints300_256/latest_ddpm-unet").to(device)
scheduler = DDPMScheduler.from_pretrained("checkpoints300256/latest_ddpm-scheduler")

model.eval()
scheduler.set_timesteps(100)
# 初始噪声
x = torch.randn(1, 3, image_size, image_size).to(device)

os.makedirs("diffusion_steps", exist_ok=True)

with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        # 预测噪声
        noise_pred = model(x, t).sample
        # 得到上一步的 x
        x = scheduler.step(noise_pred, t, x).prev_sample
        os.makedirs(f"size{image_size}", exist_ok=True)
        # 保存当前 x_t
        save_image((x + 1) / 2, f"size{image_size}/step_{i:03}.png")  # 反归一化到 [0,1]
