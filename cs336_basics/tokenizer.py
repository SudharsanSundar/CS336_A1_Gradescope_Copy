from typing import List, Dict, Tuple, Optional
import regex as re
import pprint
import time

ENCODING = 'utf-8'
ppr = pprint.PrettyPrinter()


def pretokenize_text(text) -> List[str]:
    pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = re.findall(pretoken_regex, text)

    # ppr.pprint(pretokens)

    return pretokens


def init_vocab(special_tokens) -> (Dict[int, bytes], int):
    idx = 0
    vocabulary = {}
    for token in special_tokens:
        vocabulary[idx] = token.encode(ENCODING)
        idx += 1

    for i in range(256):
        vocabulary[idx] = i.to_bytes(1, 'big')
        idx += 1

    # ppr.pprint(vocabulary)

    return vocabulary, idx


def string_to_bytes_list(input) -> List[bytes]:
    input_bytes = input.encode(ENCODING)
    input_byte_list = []

    for b in input_bytes:
        input_byte_list.append(bytes([b]))

    # print(input)
    # print(input_bytes)
    # print(input_byte_list)

    return input_byte_list


def digest_pretokens(pretokens) -> Dict[str, list]:
    pretoken_data = {}
    for token in pretokens:
        if token in pretoken_data:
            pretoken_data[token][0] += 1
        else:
            pretoken_data[token] = [1, string_to_bytes_list(token)]

    # ppr.pprint(pretoken_data)

    return pretoken_data


def count_pairs(pretoken_data) -> Dict[Tuple[bytes, bytes], int]:
    pair_counts = {}
    for pretoken in pretoken_data:
        pretoken_count = pretoken_data[pretoken][0]

        for first_tok, second_tok in zip(pretoken_data[pretoken][1][:-1], pretoken_data[pretoken][1][1:]):
            pair = (first_tok, second_tok)

            if pair in pair_counts:
                pair_counts[pair] += pretoken_count
            else:
                pair_counts[pair] = pretoken_count

    # ppr.pprint(pair_counts)

    return pair_counts


def merge_step(pair_counts, vocabulary, idx, merges) -> (Dict[int, bytes], List[Tuple[bytes, bytes]], int, bytes):
    max_freq = max(list(pair_counts.values()))
    merge_pair = max([key for key in pair_counts if pair_counts[key] == max_freq])

    merged_token = merge_pair[0] + merge_pair[1]
    vocabulary[idx] = merged_token
    merges.append(merge_pair)
    idx += 1

    # ppr.pprint(vocabulary)
    # ppr.pprint(pair_counts)
    # print(max_freq, merge_pair, merged_token)

    return vocabulary, merges, idx, merged_token


# TODO: This is probably the source of troubles
def update_pretoken_data(pretoken_data, merged_token):
    for pretoken in pretoken_data:
        pretoken_tokens = pretoken_data[pretoken][1]

        new_pretoken_tokens = []
        i = 0
        while i < len(pretoken_tokens):
            token1 = pretoken_tokens[i]

            if i < len(pretoken_tokens) - 1:
                token2 = pretoken_tokens[i + 1]

                if token1 + token2 == merged_token:
                    new_pretoken_tokens.append(merged_token)
                    i += 2
                else:
                    new_pretoken_tokens.append(token1)
                    i += 1
            else:
                new_pretoken_tokens.append(token1)
                i += 1

        pretoken_data[pretoken][1] = new_pretoken_tokens

    # ppr.pprint(pretoken_data)

    return pretoken_data


def train_tokenizer(input_path: str,
                    vocab_size: int,
                    special_tokens: List[str]):
    # Define key data var
    merges = []

    # Pretokenize the text
    text = open(input_path).read()
    pretokens = pretokenize_text(text=text)
    '''
    - where in the sequence each pretoken appears doesn't matter, since we only look within each token for pairs.
    - where in each pretoken each byte/token occurs *does* matter since pairs must be ordered, subsequent tokens in the pretoken.
    '''

    # Initialize the vocabulary with special tokens (byte encodings) and single byte values
    vocabulary, idx = init_vocab(special_tokens=special_tokens)

    # Digest pretoken data into dict of {str: [count, tokenized str in list form]}
    pretoken_data = digest_pretokens(pretokens=pretokens)

    pair_count_time = 0
    merge_step_time = 0
    update_pretoken_data_time = 0

    while len(vocabulary) < vocab_size:
        # Count the occurences of all pairs
        start = time.time()
        pair_counts = count_pairs(pretoken_data=pretoken_data)
        end = time.time()
        pair_count_time += end - start

        # Merge the most frequent pair and add it to the vocab
        start = time.time()
        vocabulary, merges, idx, merged_token = merge_step(pair_counts=pair_counts, vocabulary=vocabulary, merges=merges, idx=idx)
        end = time.time()
        merge_step_time += end - start

        # Update the pretoken data with the merged token
        start = time.time()
        pretoken_data = update_pretoken_data(pretoken_data, merged_token)
        end = time.time()
        update_pretoken_data_time += end - start

    # ppr.pprint(vocabulary)
    # ppr.pprint(pretoken_data)
    # ppr.pprint(merges)
    print('pair count time', pair_count_time, ' | merge step time', merge_step_time, '| update pretoken data time', update_pretoken_data_time)

    return vocabulary, merges


def main():
    # test_fp = 'test.txt'
    test_fp = 'speed_test.txt'
    special_tokens = ['<|endoftext|>']

    train_tokenizer(input_path=test_fp, vocab_size=500, special_tokens=special_tokens)


if __name__ == '__main__':
    main()
