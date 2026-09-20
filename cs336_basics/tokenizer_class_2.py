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

from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

ENCODING = 'utf-8'
ppr = pprint.PrettyPrinter()
VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes] = None, merges: List[Tuple[bytes, bytes]] = None, special_tokens: List[str] = None):
        self.id_to_tok = vocab
        if vocab:
            self.tok_to_id = {val: key for key, val in self.id_to_tok.items()}
        else:
            self.tok_to_id = None

        self.merges = merges
        if self.merges:
            self.merge_lookup = {pair: rank for pair, rank in zip(self.merges, range(len(self.merges)))}
        else:
            self.merge_lookup = None

        self.special_tokens = special_tokens

    def init_vocab(self, special_tokens) -> Dict[int, bytes]:
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

    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = self.init_vocab(special_tokens=special_tokens)

        with open(vocab_filepath, 'r') as f:
            vocab_added = json.load(f)

        vocab_added = {int(key): val.encode(ENCODING) for key, val in vocab_added.items()}
        vocab.update(vocab_added)

        with open(merges_filepath, 'r') as f:
            merges = json.load(f)

        merges = [(elem[0].encode(ENCODING), elem[1].encode(ENCODING)) for elem in merges]

        new_tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

        return new_tokenizer

    def pretokenize_text(self, text: str):
        if self.special_tokens:
            special_tokens = sorted(self.special_tokens, key=len)  # Short to long
        else:
            special_tokens = []

        normal_pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""      # TODO: might be better way based on Neslon's function

        def split_text(block: str, remaining_tokens: List[str]):
            if len(remaining_tokens) == 0:
                return re.findall(normal_pretoken_regex, block)
            else:
                current_token = remaining_tokens[-1]
                sub_blocks = block.split(current_token)
                final_tokens = []

                for sub_block in sub_blocks:
                    if len(sub_block) > 0:
                        final_tokens += split_text(sub_block, remaining_tokens[:-1])
                    final_tokens += [current_token]

                return final_tokens[:-1]

        all_final_tokens = split_text(text, special_tokens)

        return all_final_tokens

    def string_to_bytes_list(self, input) -> List[bytes]:
        input_bytes = input.encode(ENCODING)
        input_byte_list = []

        for b in input_bytes:
            input_byte_list.append(bytes([b]))

        return input_byte_list

    def merge_tokens(self, pretokens: List[str]) -> List[bytes]:
        final_tokens = []
        for pretoken in pretokens:
            if self.special_tokens and pretoken in self.special_tokens:
                final_tokens.append(pretoken.encode(ENCODING))
                continue

            pretoken_bytes = self.string_to_bytes_list(pretoken)

            old_pretoken_bytes = pretoken_bytes
            keep_merging = True

            while keep_merging:
                next_merge = [len(self.id_to_tok) + 1, ()]

                for first_tok, second_tok in zip(old_pretoken_bytes[:-1], old_pretoken_bytes[1:]):
                    pair = (first_tok, second_tok)

                    if pair in self.merge_lookup:
                        if self.merge_lookup[pair] < next_merge[0]:
                            next_merge = [self.merge_lookup[pair], pair]

                if next_merge[0] == len(self.id_to_tok) + 1:
                    keep_merging = False
                else:
                    counter = 0
                    new_pretoken_bytes = []

                    while counter < len(old_pretoken_bytes):
                        first_tok = old_pretoken_bytes[counter]

                        if counter < len(old_pretoken_bytes) - 1:
                            second_tok = old_pretoken_bytes[counter + 1]

                            if (first_tok, second_tok) == next_merge[1]:
                                new_pretoken_bytes.append(first_tok + second_tok)
                                counter += 2
                            else:
                                new_pretoken_bytes.append(first_tok)
                                counter += 1
                        elif counter == len(old_pretoken_bytes) - 1:
                            new_pretoken_bytes.append(first_tok)
                            counter += 1

                    old_pretoken_bytes = new_pretoken_bytes

            final_tokens += old_pretoken_bytes

        return final_tokens

    def map_tokens_to_ids(self, tokens: List[bytes]) -> List[int]:
        ids = []

        for token in tokens:
            ids.append(self.tok_to_id[token])

        return ids

    def encode(self, text: str) -> List[int]:
        start = time.time()

        # First, pretokenize text
        pretokens = self.pretokenize_text(text=text)

        # Second, merge bytes within pretokens according to tokenizer's sequence of merges
        final_tokens = self.merge_tokens(pretokens=pretokens)

        # Finally, map bytes to token ids
        token_ids = self.map_tokens_to_ids(tokens=final_tokens)

        end = time.time()
        print('Total time taken (seconds):', end - start)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(text=line)

    def decode(self, ids: List[int]) -> str:
        all_bytes = b''
        for b in [self.id_to_tok[id] for id in ids]:
            all_bytes += b

        decoded_str = all_bytes.decode(encoding=ENCODING, errors='replace')

        return decoded_str


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
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


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


def sample_ten_docs_ts(fp):
    samples = ''
    with open(fp, 'r') as f:
        counter = 0
        for line in f:
            samples += line
            if '<|endoftext|>' in line:
                counter += 1
            if counter == 10:
                break

    return samples


def sample_ten_docs_owt(fp):
    samples = ''
    with open(fp, 'r') as f:
        counter = 0
        for line in f:
            samples += line
            if '<|endoftext|>' in line:
                counter += 1
            if counter == 10:
                break

    return '<|endoftext|>'.join(samples.split('<|endoftext|>')[:10]) + '<|endoftext|>'


def main():
    ts_tokker = get_tokenizer_from_vocab_merges_path(vocab_path='../../../data/TinyStoriesV2-GPT4-tokenizer/vocab.json', merges_path='../../../data/TinyStoriesV2-GPT4-tokenizer/merges.bpe', special_tokens=['<|endoftext|>'])
    owt_tokker = get_tokenizer_from_vocab_merges_path(vocab_path='../../../data/owt-tokenizer/vocab.json', merges_path='../../../data/owt-tokenizer/merges.bpe', special_tokens=['<|endoftext|>'])

    ts = '../../../data/TinyStoriesV2-GPT4-valid.txt'
    owt = '../../../data/owt_valid.txt'
    ts_samples = sample_ten_docs_ts(ts)
    owt_samples = sample_ten_docs_owt(owt)

    ts_encoded = ts_tokker.encode(text=ts_samples)
    owt_encoded = ts_tokker.encode(text=owt_samples)
    owt_owt = owt_tokker.encode(text=owt_samples)

    print('ts ts compression:', len(ts_samples.encode(ENCODING)), '/', len(ts_encoded), '=', len(ts_samples.encode(ENCODING))/len(ts_encoded))
    print('ts owt compression:', len(owt_samples.encode(ENCODING)), '/', len(owt_encoded), '=', len(owt_samples.encode(ENCODING)) / len(owt_encoded))
    print('owt owt compression:', len(owt_samples.encode(ENCODING)), '/', len(owt_owt), '=', len(owt_samples.encode(ENCODING)) / len(owt_owt))


if __name__ == '__main__':
    main()
