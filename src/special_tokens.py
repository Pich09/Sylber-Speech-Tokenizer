"""Special speech-token IDs (pad/bos/eos), derived from each encoder's own
fitted k-means vocabulary size rather than a fixed config constant.

Different encoders can land on different K (discretization.py sweep/fit runs
independently per --encoder), so a single hardcoded {pad: 10000, bos: 10001,
eos: 10002} — valid only when K==10000 — silently produces out-of-range
token IDs (and an out-of-bounds embedding lookup at train time) for any
encoder fit to a different K. Always compute these from the actual vocab
size instead.
"""
from __future__ import annotations

SPECIAL_TOKEN_NAMES = ("pad", "bos", "eos")


def special_token_ids(vocab_size: int) -> dict[str, int]:
    """IDs are placed immediately after the k-means vocabulary: vocab_size,
    vocab_size+1, vocab_size+2 for pad/bos/eos respectively."""
    return {name: vocab_size + i for i, name in enumerate(SPECIAL_TOKEN_NAMES)}
