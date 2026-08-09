# train_ixi.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append("F:/IXI/SNRAware-main")

import torch
from omegaconf import OmegaConf
from src.snraware.projects.mri.denoising.model import DenoisingModel
from dataset import IXIDataset
from model import SimpleUNet


cfg = Config()

device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_set = IXIDataset(cfg.train_input, cfg.train_target)
loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)

model = SimpleUNet(cfg.in_channels, cfg.out_channels, cfg.features).to(device)

opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
loss_fn = nn.MSELoss()

for epoch in range(cfg.epochs):

    model.train()
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch+1}: Loss {total/len(loader):.6f}")

torch.save(model.state_dict(), "snraware_stable.pth")
print("Training Done")