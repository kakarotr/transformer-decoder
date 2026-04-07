import torch
import torch.nn.functional as F


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index=-100):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1).long(),
        ignore_index=ignore_index,
    )
