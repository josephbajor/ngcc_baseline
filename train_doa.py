import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torch.utils.data as data_utils
from tqdm import tqdm
import numpy as np
import random
from torchinfo import summary
import argparse
import os
from params import get_params

from model import NGCCPHAT, PGCCPHAT, GCC
from data_doa import build_loaders
from helpers import LabelSmoothing
import cfg

# Import params
params = get_params()

# for reproducibility
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)

max_tau_gcc = int(
    np.floor(
        np.linalg.norm(cfg.mic_locs_train[:, 0] - cfg.mic_locs_train[:, 1])
        * cfg.fs
        / 343
    )
)

sig_len = cfg.sig_len
epochs = cfg.epochs
batch_size = cfg.batch_size
lr = cfg.lr
fs = cfg.fs
max_tau = cfg.max_delay
wd=cfg.wd

train_loader, val_loader, test_loader = build_loaders(params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

# load model
if cfg.model == "NGCCPHAT":
    use_sinc = True if not cfg.no_sinc else False
    model = NGCCPHAT(max_tau, cfg.head, use_sinc, sig_len, cfg.num_channels, fs)
elif cfg.model == "PGCCPHAT":
    model = PGCCPHAT(max_tau=max_tau_gcc, head=cfg.head)
else:
    raise Exception("Please specify a valid model")

model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    train_loss = 0

    model.train()

    pbar_update = batch_size
    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, (x1, x2, doa) in enumerate(train_loader):
            bs = x1.shape[0]

            x1 = x1.to(device)
            x2 = x2.to(device)
            doa = doa.to(device)
            y_hat = model(x1, x2)

            loss = loss_fn(y_hat, doa)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item() * bs

            pbar.update(pbar_update)

    train_loss = train_loss / len(train_loader)

    outstr = (
        "Train epoch %d, loss: %.6f"
        % (epoch, train_loss)
    )

    scheduler.step()

    torch.cuda.empty_cache()

    # Validation
    model.eval()

    val_loss = 0.0
    with tqdm(total=len(val_loader)) as pbar:
        for batch_idx, (x1, x2, doa) in enumerate(val_loader):
            with torch.no_grad():
                bs = x1.shape[0]
                x1 = x1.to(device)
                x2 = x2.to(device)
                doa = doa.to(device)
                y_hat = model(x1, x2)

                loss = loss_fn(y_hat, doa)
                val_loss += loss.detach().item() * bs

                pbar.update(pbar_update)

    val_loss = val_loss / len(val_loader)

    outstr = (
        "Val epoch %d, loss: %.6f"
        % (epoch, val_loss)
    )

    torch.cuda.empty_cache()
