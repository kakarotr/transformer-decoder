import torch
import torch.nn.functional as F


def compute_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
):
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.view(-1)

    if not flat_labels.ne(ignore_index).any():
        return flat_hidden.sum() * 0.0

    flat_logits = F.linear(flat_hidden.float(), lm_head_weight.float())

    return F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
    )


@torch.no_grad()
def eval_compute_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
):
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.view(-1)

    flat_logits = F.linear(flat_hidden.float(), lm_head_weight.float())

    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="sum",
    )
    valid_token_count = flat_labels.ne(ignore_index).sum()
    return loss, valid_token_count
