import torch
import torch.nn.functional as F


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index=-100):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
    flat_labels = shift_labels.view(-1).long()

    if not flat_labels.ne(ignore_index).any():
        return flat_logits.sum() * 0.0

    return F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
    )


@torch.no_grad()
def eval_compute_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index=-100):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
    flat_labels = shift_labels.view(-1).long()

    valid_token_count = flat_labels.ne(ignore_index).sum()
    if valid_token_count.item() == 0:
        return flat_logits.sum() * 0.0, valid_token_count

    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="sum",
    )
    return loss, valid_token_count
