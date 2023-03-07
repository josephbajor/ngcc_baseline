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

from model import NGCCPHAT, PGCCPHAT, GCC
from data import (
    LibriSpeechLocations,
    DelaySimulator,
    one_random_delay,
    remove_silence,
    SimData,
)
from helpers import LabelSmoothing, display_test_results_NGCC
import cfg

from params import get_params

args = get_params()

# Librispeech dataset constants
DATA_LEN = 2620
VAL_IDS = [260, 672, 908]  # use these speaker ids for validation
TEST_IDS = [61, 121, 237]  # use these speaker ids for testing
NUM_TEST_WINS = 15
MIN_SIG_LEN = 2  # only use snippets longer than 2 seconds

# for reproducibility
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)

# calculate the max_delay for gcc
max_tau_gcc = int(
    np.floor(
        np.linalg.norm(cfg.mic_locs_train[:, 0] - cfg.mic_locs_train[:, 1])
        * cfg.fs
        / 343
    )
)

# training parameters
max_tau = cfg.max_delay
snr = cfg.snr
t60 = cfg.t60
fs = cfg.fs
sig_len = cfg.sig_len
epochs = cfg.epochs
batch_size = cfg.batch_size
lr = cfg.lr
wd = cfg.wd
label_smooth = cfg.ls

test_set = SimData(cfg.sim_data_path, set_type="test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

if cfg.model == "NGCCPHAT":
    use_sinc = True if not cfg.no_sinc else False
    model = NGCCPHAT(max_tau, cfg.head, use_sinc, sig_len, cfg.num_channels, fs)
elif cfg.model == "PGCCPHAT":
    model = PGCCPHAT(max_tau=max_tau_gcc, head=cfg.head)
else:
    raise Exception("Please specify a valid model")

model = model.to(device)
model.eval()
summary(model, [(1, 1, sig_len), (1, 1, sig_len)])

gcc = GCC(max_tau=max_tau_gcc)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

if cfg.loss == "ce":
    loss_fn = LabelSmoothing(label_smooth)
elif cfg.loss == "mse":
    loss_fn = nn.MSELoss()
else:
    raise Exception("Please specify a valid loss function")

test_len = len(test_set)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

model.load_state_dict(
    torch.load(
        "experiments/" + args.exp_name + "/model.pth",
        map_location=torch.device(device),
    )
)

model.eval()

LOG_DIR = os.path.join("experiments/" + args.exp_name + "/")
if cfg.anechoic:
    name = "eval_anechoic.txt"
else:
    name = "eval.txt"
LOG_FOUT = open(os.path.join(LOG_DIR, name), "w")
LOG_FOUT.write(str(args) + "\n")

if cfg.anechoic:
    t60_range = [0.0]
else:
    t60_range = cfg.t60_range

pbar_update = batch_size

test_angles, test_rt60s, test_delays, test_snr, test_preds, test_preds_gcc = (
    [],
    [],
    [],
    [],
    [],
    [],
)

test_loss = 0
with tqdm(total=test_len) as pbar:
    for batch_idx, (x1, x2, angle, rt60, delays, snr) in enumerate(test_loader):
        with torch.no_grad():
            bs = x1.shape[0]
            x1 = x1.type(torch.FloatTensor).to(device)
            x2 = x2.type(torch.FloatTensor).to(device)
            delays = delays.to(device)

            # win_len = x1.shape[-1]//NUM_TEST_WINS
            x1 = x1.unfold(-1, sig_len, sig_len).flatten(0,1)
            x2 = x2.unfold(-1, sig_len, sig_len).flatten(0,1)

            y_hat = model(x2, x1)

            cc = gcc(x1.squeeze(), x2.squeeze())
            shift_gcc = torch.argmax(cc, dim=-1) - max_tau_gcc

            delays_loss = torch.round(delays).type(torch.LongTensor)
            shift = torch.argmax(y_hat, dim=-1) - max_tau

            test_angles.append(angle)
            test_rt60s.append(rt60)
            test_delays.append(delays)
            test_snr.append(snr)
            test_preds.append(shift)
            test_preds_gcc.append(shift_gcc)

            # loss = loss_fn(y_hat, delays_loss.to(device))
            # test_loss += loss * bs

            pbar.update(pbar_update)

# print(f'test loss: {test_loss/test_len}')

# # Calculate proper rt60 bins
# bin_size = 0.1
# min_rt60 = test_rt60s.min()
# max_rt60 = test_rt60s.max()

# rt60_bins = np.linspace()

display_test_results_NGCC(
    args,
    delays=test_delays,
    preds=test_preds,
    preds_gcc=test_preds_gcc,
    rt60s=test_rt60s,
    snrs=test_snr,
    max_tau=max_tau,
    t = cfg.t,
)