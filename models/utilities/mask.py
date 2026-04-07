from typing import overload

import torch


def create_causal_mask(
    seq_len: int,
    *,
    device: torch.device | None = None,
):
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    mask = mask.tril()
    return mask.unsqueeze(0).unsqueeze(0)


@overload
def create_padding_mask(
    *,
    input_ids: torch.Tensor,
    pad_token_id: int,
    device: torch.device | None = None,
) -> torch.Tensor: ...


@overload
def create_padding_mask(
    *,
    attention_mask: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor: ...


def create_padding_mask(
    *,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    pad_token_id: int | None = None,
    device: torch.device | None = None,
):
    use_input_ids = input_ids is not None
    use_attention_mask = attention_mask is not None

    if use_input_ids == use_attention_mask:
        raise ValueError("Exactly one of input_ids or attention_mask must be provided")
    if use_input_ids:
        if pad_token_id is None:
            raise ValueError("pad_token_id must not be None when input_ids is provided")
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D [batch_size, seq_len], got {tuple(input_ids.shape)}")

        mask = (input_ids != pad_token_id).to(device)
        return mask.unsqueeze(1).unsqueeze(1)

    assert attention_mask is not None
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be 2D [batch_size, seq_len], got {tuple(attention_mask.shape)}")

    mask = attention_mask != 0
    mask = mask.to(device)
    return mask.unsqueeze(1).unsqueeze(1)
