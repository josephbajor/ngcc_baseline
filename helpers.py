import torch
import torch.nn as nn
import time
import wandb


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def initiate_run(params):
    """
    Initialize connection to wandb and begin the run using provided hparams
    """
    with open(params.keyring_root + "wandb.key") as key:
        wandb.login(key=key.read().strip())
        key.close()

    if params.debug:
        mode = "disabled"
    else:
        mode = "online"

    run = wandb.init(
        name=f"{params.run_name}_{int(time.time())}",
        project="NGCC_Baseline_DOA",
        mode=mode,
    )

    return run

def display_test_results_NGCC(params, delays=None, preds=None, preds_gcc=None, rt60s=None, snrs=None, max_tau=None, t=None):

    max_lag = params.sep_len/343*params.sample_rate
    loss = torch.nn.functional.l1_loss(preds, delays)
    loss_lags = torch.sin(loss)*max_lag

    # print('Overall error: {:.3f} degrees\t{:.3f} lags'.format(loss*180/torch.pi, loss_lags))
    # print('')
    print('RT60\t|\tRMSE\t|\tGCC RMSE\t|\tMAE\t|\tGCC MAE\t|\tACC\t|\tGCC ACC')
    print('-'*120)

    reverb_list = torch.linspace(0.3, 1.0, 8)

    for ii,rt60 in enumerate(reverb_list):

        if ii == 0:
            idxs = rt60s<rt60
        else:
            idxs = torch.logical_and(rt60s<rt60, rt60s>reverb_list[ii-1])
        rt60 -= 0.1

        shift = preds[idxs]
        gt = delays[idxs]
        shift_gcc = preds_gcc[idxs]
        # reverb_loss = torch.nn.functional.l1_loss(reverb_preds, reverb_targets)
        # reverb_loss_lag = torch.sin(reverb_loss)*max_lag

        rmse = torch.sqrt(torch.sum(torch.abs(shift-gt)**2)/len(idxs))
        gcc_rmse = torch.sqrt(torch.sum(torch.abs(shift_gcc-gt)**2)/len(idxs))
        mae = torch.sum(torch.abs(shift-gt))/len(idxs)
        gcc_mae = torch.sum(torch.abs(shift_gcc-gt))/len(idxs)
        acc = torch.sum(torch.abs(shift - gt) < t)/len(idxs)
        gcc_acc = torch.sum(torch.abs(shift_gcc - gt) < t)/len(idxs)


        print('{:.2f}\t|\t{:.3f}\t|\t{:.3f}\t|\t{:.3f}\t|\t{:.3f}\t|\t{:.3f}\t|\t{:.3f}'.format(rt60, rmse, gcc_rmse, mae, gcc_mae, acc, gcc_acc))

    snr_list = torch.linspace(torch.floor(snrs.min()), torch.ceil(snrs.max()), 6)[1:]
    print('')
    print('SNR\t|\tRMSE\t|\tGCC RMSE\t|\tMAE\t|\tGCC MAE\t|\tACC\t|\tGCC ACC')
    print('-'*120)
    for ii,snr in enumerate(snr_list):
        if ii == 0:
            idxs = snrs<snr
        else:
            idxs = torch.logical_and(snrs<snr, snrs>snr_list[ii-1])

        shift = preds[idxs]
        gt = delays[idxs]
        shift_gcc = preds_gcc[idxs]
        # reverb_loss = torch.nn.functional.l1_loss(reverb_preds, reverb_targets)
        # reverb_loss_lag = torch.sin(reverb_loss)*max_lag

        rmse = torch.sqrt(torch.sum(torch.abs(shift-gt)**2)/len(idxs))
        gcc_rmse = torch.sqrt(torch.sum(torch.abs(shift_gcc-gt)**2)/len(idxs))
        mae = torch.sum(torch.abs(shift-gt))/len(idxs)
        gcc_mae = torch.sum(torch.abs(shift_gcc-gt))/len(idxs)
        acc = torch.sum(torch.abs(shift - gt) < t)/len(idxs)
        gcc_acc = torch.sum(torch.abs(shift_gcc - gt) < t)/len(idxs)

        print('{:.2f}\t|\t{:.3f}\t|\t{:.3f}\t|\t{:.3f}\t|\t{:.3f}\t|\t{:.3f}\t|\t{:.3f}'.format(rt60, rmse, gcc_rmse, mae, gcc_mae, acc, gcc_acc))