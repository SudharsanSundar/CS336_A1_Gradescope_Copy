from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Iterable, Iterator
import regex as re
import pprint
import time
import sys
import json

import json
import os
import resource
import sys
from typing import Optional

import pytest
import tiktoken

from itertools import chain

import resource
import tracemalloc

from cs336_basics import tokenizer_class_2
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

ENCODING = 'utf-8'
ppr = pprint.PrettyPrinter()
VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: Optional[list[str]] = None,
):
    """Given the path to a JSON vocab, a file with BPE merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab: dict[int, bytes]
            The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges: list[tuple[bytes, bytes]]
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens: Optional[list[str]]
            A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return tokenizer_class_2.Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: Optional[list[str]] = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


def get_vocab(
    vocab_path: str,
    special_tokens=None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)

    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    return vocab


def get_longest_token(vocab):
    longest = ''
    aux = []
    for key in vocab:
        candidate = vocab[key]
        if len(candidate) > len(longest):
            longest = candidate
            aux = []
        elif len(candidate) == len(longest):
            aux.append(candidate)

    print('longest:', longest)
    print('equally long:', aux)


def init_vocab(special_tokens) -> Dict[int, bytes]:
    idx = 0
    vocabulary = {}
    if special_tokens:
        for token in special_tokens:
            vocabulary[idx] = token.encode(ENCODING)
            idx += 1

    for i in range(256):
        vocabulary[idx] = i.to_bytes(1, 'big')
        idx += 1

    return vocabulary


def check_equivalent_vocabs(fp1, fp2):
    vocab1 = json.load(open(fp1, 'r'))
    vocab1 = {int(key): value.encode(ENCODING) for key, value in vocab1.items()}
    vocab_prefix = init_vocab(['<|endoftext|>'])
    vocab1.update(vocab_prefix)
    vocab2 = get_vocab(fp2, special_tokens=['<|endoftext|>'])

    print('same length?', len(vocab1), len(vocab2))

    # ppr.pprint(vocab1)

    total_diff1 = 0

    for value in vocab2.values():
        if value not in vocab1.values():
            total_diff1 += 1
            print('DIFF!', value, 'in vocab2, not in vocab1')

    total_diff2 = 0

    for value in vocab1.values():
        if value not in vocab2.values():
            total_diff2 += 1
            print('DIFF!', value, 'in vocab1, not in vocab2')

    print('total diffs', total_diff1, total_diff2)


def compare_tokenizers(voc1, voc2):
    for pair in zip(list(voc1.items())[-500:], list(voc2.items())[9500:10000]):
        ppr.pprint(pair)
        print('--')

    print('avg lens', sum([len(val) for val in list(voc1.values())]) / len(voc1), sum([len(val) for val in list(voc2.values())[:10000]]) / 10000)


def main():
    fp1 = './trained_vocab_owt.json'
    fp2 = '../../../data/owt-tokenizer/vocab.json'
    fp3 = '../../../data/TinyStoriesV2-GPT4-tokenizer/vocab.json'

    check_equivalent_vocabs(fp1, fp2)

    vocab1 = json.load(open(fp1, 'r'))
    vocab1 = {int(key): value.encode(ENCODING) for key, value in vocab1.items()}
    vocab_prefix = init_vocab(['<|endoftext|>'])
    vocab1.update(vocab_prefix)

    vocab2 = get_vocab(fp2, special_tokens=['<|endoftext|>'])
    vocab3 = get_vocab(fp3, special_tokens=['<|endoftext|>'])

    get_longest_token(vocab2)
    compare_tokenizers(vocab2, vocab3)


if __name__ == '__main__':
    main()
