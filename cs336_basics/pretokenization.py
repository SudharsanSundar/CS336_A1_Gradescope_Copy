from typing import List, Dict, Tuple, Optional, Iterable, Iterator
import regex as re
import pprint
import time
import sys
import json

ENCODING = 'utf-8'
ppr = pprint.PrettyPrinter()

special_tokens = ['<|endoftext|>']


def pretok_test(text: str):
    sp_tok = '<|endoftext|>'
    regex = r"\<\|endoftext\|\>|[^<](?!\|endoftext\|\>)+"
    toks = re.findall(regex, text)

    print(text)
    print(toks)

    print(text.split(sp_tok))


def pretok_naive(text: str):
    sp_tok = '<|endoftext|>'
    pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    all_toks = []

    print('orig:', text)
    print('blocks:', text.split(sp_tok))
    for block in text.split(sp_tok):
        if len(block) > 0:
            print(re.findall(pretoken_regex, block))
            all_toks += re.findall(pretoken_regex, block)
        all_toks += [sp_tok]

    all_toks = all_toks[:-1]

    print('all toks:', all_toks)

    return all_toks


def pretokenize(text: str):
    special_token_list = ['<endoftext>', '<endoftext><endoftext>']
    # special_token_list = ['<endoftext>']
    special_tokens = sorted(special_token_list, key=len)     # Short to long
    normal_pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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


def main():
    text1 = open('special_text.txt').read()
    text2 = open('test.txt').read()

    # pretok_test(text1)
    # pretok_naive(text2)

    print(pretokenize(text2))


if __name__ == '__main__':
    main()

























# def pretokenize_text(self, text: str) -> List[str]:
    #     # special_tokens_regex = r""""""
    #     # for special_token in self.special_tokens:
    #     #     special_tokens_regex += r"""(""" + repr(special_token) + r"""){1}|"""
    #
    #     # special_tokens_regex = r"""(?=(\<\|endoftext\|\>))|"""
    #     # print(repr(special_tokens_regex))
    #     all_things = []
    #     for special_token in self.special_tokens:
    #         all_things += self.partition_special_tokens(special_token)
    #
    #
    #     # pretoken_regex = special_tokens_regex
    #     # pretoken_regex = special_tokens_regex + r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #     pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #
    #     print(pretoken_regex)
    #
    #     pretokens = re.findall(pretoken_regex, text)
    #
    #     print(pretokens)
    #
    #     return pretokens

    # def pretok_naive(self, text: str):
    #     sp_tok = '<|endoftext|>'
    #     pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #     all_toks = []
    #
    #     # print('orig:', text)
    #     # print('blocks:', text.split(sp_tok))
    #     for block in text.split(sp_tok):
    #         if len(block) > 0:
    #             # print(re.findall(pretoken_regex, block))
    #             all_toks += re.findall(pretoken_regex, block)
    #         all_toks += [sp_tok]
    #
    #     all_toks = all_toks[:-1]
    #
    #     # print('all toks:', all_toks)
    #
    #     return all_toks









# def digest_pretokens(pretokens, special_tokens) -> Dict[str, list]:
#     pretoken_data = {}
#     for token in pretokens:
#         if token not in special_tokens and token[1:] not in special_tokens and token[:-1] not in special_tokens and token[1:-1] not in special_tokens:
#             if token in pretoken_data:
#                 pretoken_data[token][0] += 1
#             else:
#                 pretoken_data[token] = [1, string_to_bytes_list(token)]
#
#     # ppr.pprint(pretoken_data)
#
#     return pretoken_data