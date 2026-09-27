from __future__ import annotations

import unittest

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import XLMRobertaTokenizerFast

from app.classifier_input import encode_classifier_texts
from train_classifier import make_loader


def toy_tokenizer() -> XLMRobertaTokenizerFast:
    vocabulary = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    vocabulary.update({f"w{index}": index + 4 for index in range(12)})
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 2), ("</s>", 3)])
    return XLMRobertaTokenizerFast(tokenizer_object=raw, unk_token="<unk>",
                                   pad_token="<pad>", cls_token="<s>", sep_token="</s>")


class ClassifierInputTests(unittest.TestCase):
    def test_head_tail_preserves_both_ends_and_special_tokens(self) -> None:
        tokenizer = toy_tokenizer()
        text = " ".join(f"w{index}" for index in range(10))
        head = encode_classifier_texts(tokenizer, [text], max_length=8, strategy="head-8",
                                       pad_to_max_length=True)
        tail = encode_classifier_texts(tokenizer, [text], max_length=8, strategy="head-tail-8",
                                       pad_to_max_length=True)
        self.assertEqual(head["input_ids"].tolist(), [[2, 4, 5, 6, 7, 8, 9, 3]])
        self.assertEqual(tail["input_ids"].tolist(), [[2, 4, 5, 6, 11, 12, 13, 3]])
        self.assertEqual(tail["attention_mask"].tolist(), [[1] * 8])

        loader = make_loader([{"text": text, "topic_id": "roads"}], tokenizer, 1, 8,
                             False, "head-tail")
        self.assertEqual(next(iter(loader))[0].tolist(), tail["input_ids"].tolist())

    def test_short_text_is_unchanged_and_artifact_strategy_must_match_length(self) -> None:
        tokenizer = toy_tokenizer()
        encoded = encode_classifier_texts(tokenizer, ["w0 w1"], max_length=8,
                                          strategy="head-tail-8", pad_to_max_length=True)
        self.assertEqual(encoded["input_ids"].tolist(), [[2, 4, 5, 3, 0, 0, 0, 0]])
        self.assertEqual(encoded["attention_mask"].tolist(), [[1, 1, 1, 1, 0, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "unsupported classifier input length strategy"):
            encode_classifier_texts(tokenizer, ["w0"], max_length=8,
                                    strategy="head-tail-16", pad_to_max_length=False)


if __name__ == "__main__":
    unittest.main()
