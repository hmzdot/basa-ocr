import torch


def ctc_greedy_decode(x, max_letters: int, blank_token=0):
    """
    Converts a sequence like H_ELL_L_OO_ into HELLO______

    In: x (B, T)
    Out: (B, max_len)
    """
    B, T = x.shape
    out = torch.full(
        (B, max_letters),
        blank_token,
        dtype=torch.long,
        device=x.device,
    )

    for b in range(B):
        decoded = []
        prev_char = blank_token

        for t in range(T):
            cur_char = x[b, t].item()
            if cur_char != blank_token and cur_char != prev_char:
                decoded.append(cur_char)
            prev_char = cur_char

        for i, c in enumerate(decoded[:max_letters]):
            out[b, i] = c

    return out
