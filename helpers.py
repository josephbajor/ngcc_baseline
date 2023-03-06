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