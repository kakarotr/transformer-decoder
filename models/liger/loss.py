import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

loss_fn = LigerFusedLinearCrossEntropyLoss()


def compute_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
):
    shift_hidden_states = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
    flat_labels = shift_labels.view(-1)

    if not flat_labels.ne(ignore_index).any():
        return flat_hidden_states.sum() * 0.0

    weight = lm_head_weight.to(dtype=flat_hidden_states.dtype)
    return loss_fn(weight, flat_hidden_states, flat_labels, ignore_index=ignore_index)


@torch.no_grad()
def eval_compute_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
):
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_hidden_states = shift_hidden.view(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.view(-1)

    valid_token_count = flat_labels.ne(ignore_index).sum()

    if valid_token_count.item() == 0:
        return flat_hidden_states.sum() * 0.0, valid_token_count

    weight = lm_head_weight.to(dtype=flat_hidden_states.dtype)
    mean_loss = loss_fn(
        weight,
        flat_hidden_states,
        flat_labels,
        ignore_index=ignore_index,
    )

    loss_sum = mean_loss * valid_token_count
    return loss_sum, valid_token_count
