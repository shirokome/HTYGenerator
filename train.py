# train_ddpm.py
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

# ====== configs ======
image_size = 256
batch_size = 2
timesteps = 2000
epochs = 3000
device = "cuda:6"

# ====== model & scheduler ======
model = UNet2DModel(
    sample_size=image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=3,
    block_out_channels=(256, 512, 1024),
    down_block_types=("DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D"),
).to(device)

scheduler = DDPMScheduler(num_train_timesteps=timesteps)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
latest_unet_path = "/home/xuanran/htyGenerator/checkpoints300_256/latest_ddpm-unet"
latest_scheduler_path="/home/xuanran/htyGenerator/checkpoints300256/latest_ddpm-scheduler"
checkpoints300_256 = "/home/xuanran/htyGenerator/checkpoints300_256"
if os.path.exists(latest_unet_path):
    print(f"🔄 Loading checkpoint from {latest_unet_path}")
    model = UNet2DModel.from_pretrained(latest_unet_path).to(device)
    scheduler = DDPMScheduler.from_pretrained(latest_scheduler_path)
    
    # 自动识别上次训练到第几轮
    existing_epochs = [
        int(name.split("ddpm-unet")[-1].split("_loss")[0])
        for name in os.listdir(checkpoints300_256)
        if name.startswith("ddpm-unet") and "_loss" in name
    ]
    if existing_epochs:
        start_epoch = max(existing_epochs) + 1
        print(f"⏩ Resuming from epoch {start_epoch}")
    else:
        print("⚠️ Could not determine previous epoch, starting from 0.")
        start_epoch = 0
# ====== dataset ======
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

dataset = datasets.ImageFolder(root="./photos", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ====== training ======
model.train()
print(f"Training on {len(dataset)} images with batch size {batch_size} for {epochs} epochs.")
print(f"Using device: {device}")
print(f"Model: {model.__class__.__name__}")
print(f"Scheduler: {scheduler.__class__.__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):.3e}")
for epoch in range(start_epoch, epochs):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    total_loss = 0
    for step, (x, _) in enumerate(pbar):
        x = x.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(0, timesteps, (x.size(0),), device=device).long()
        noisy_x = scheduler.add_noise(x, noise, t)
        noise_pred = model(noisy_x, t).sample
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        os.makedirs(f"checkpoints{epochs}_{image_size}", exist_ok=True)
        model.save_pretrained(f"checkpoints{epochs}_{image_size}/ddpm-unet{epoch}_loss{avg_loss}")
        scheduler.save_pretrained(f"checkpoints{epochs}_{image_size}/ddpm-scheduler{epoch}_loss{avg_loss}")
        model.save_pretrained(f"checkpoints{epochs}_{image_size}/latest_ddpm-unet")
        scheduler.save_pretrained(f"checkpoints{epochs}_{image_size}/latest_ddpm-scheduler")
