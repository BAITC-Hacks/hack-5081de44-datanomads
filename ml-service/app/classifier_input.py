"""Apply the classifier artifact's input length strategy consistently."""

from __future__ import annotations


def encode_classifier_texts(tokenizer, texts: list[str], *, max_length: int,
                            strategy: str, pad_to_max_length: bool):
    if strategy == f"head-{max_length}":
        return tokenizer(texts, truncation=True, max_length=max_length,
                         padding="max_length" if pad_to_max_length else False,
                         return_tensors="pt")
    if strategy != f"head-tail-{max_length}":
        raise ValueError("unsupported classifier input length strategy")

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    token_budget = max_length - special_tokens
    if token_budget < 2:
        raise ValueError("classifier max length is too short for head-tail encoding")
    raw_ids = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    sequences = []
    for ids in raw_ids:
        if len(ids) > token_budget:
            head_count = (token_budget + 1) // 2
            tail_count = token_budget - head_count
            ids = ids[:head_count] + ids[-tail_count:]
        sequence = tokenizer.build_inputs_with_special_tokens(ids)
        if len(sequence) != len(ids) + special_tokens:
            raise ValueError("classifier tokenizer did not add the expected special tokens")
        sequences.append(sequence)
    return tokenizer.pad({"input_ids": sequences},
                         padding="max_length" if pad_to_max_length else True,
                         max_length=max_length if pad_to_max_length else None,
                         return_tensors="pt")
