from typing import List, Dict, Tuple, Optional
import regex as re
import pprint

ENCODING = 'utf-8'
ppr = pprint.PrettyPrinter()


def initialize_vocab(special_tokens: List[str]):
    vocabulary: Dict[int, bytes] = {}
    idx = 0
    for special_token in special_tokens:
        vocabulary[idx] = special_token.encode(ENCODING)
        idx += 1

    for i in range(256):
        vocabulary[idx] = i.to_bytes(1, 'big')
        idx += 1

    return vocabulary, idx


def get_dict_keys_with_max_vals(input_dict):
    max_val = max(input_dict.values())
    return [key for key in input_dict if input_dict[key] == max_val]


def get_pair_counts(token_bytes):
    neighbors = {}

    for i in range(0, len(token_bytes) - 1, 1):
        byte1 = bytes([token_bytes[i]])
        byte2 = bytes([token_bytes[i + 1]])
        if i == 0:
            pair_type = 'start'
        elif i == len(token_bytes) - 2:
            pair_type = 'end'
        else:
            pair_type = 'middle'
        pair = ((byte1, byte2), pair_type)

        if pair in neighbors:
            neighbors[pair] += 1
        else:
            neighbors[pair] = 1

    return neighbors


def get_total_pair_counts(pretoken_counts):
    total_pair_counts = {}

    for token_bytes in pretoken_counts:
        for pair in pretoken_counts[token_bytes]['pair_counts']:
            pair_vals = pair[0]
            if pair_vals in total_pair_counts:
                total_pair_counts[pair_vals] += pretoken_counts[token_bytes]['count'] * \
                                           pretoken_counts[token_bytes]['pair_counts'][pair]
            else:
                total_pair_counts[pair_vals] = pretoken_counts[token_bytes]['count'] * \
                                          pretoken_counts[token_bytes]['pair_counts'][pair]

    return total_pair_counts


def propagate_new_merge(pretoken_counts, merge_pair, merged_token):
    # Update indexed token counts with new merged token
    for token_bytes in pretoken_counts:
        if merge_pair in pretoken_counts[token_bytes]['pair_counts']:
            del pretoken_counts[token_bytes]['pair_counts'][merge_pair]

            new_entries = []
            del_pairs = []
            # TODO: this for is still looking at every pair within the pretoken
            for pair in pretoken_counts[token_bytes]['pair_counts']:
                if pair[1] == 'middle':
                    pair_vals = pair[0]
                    if merge_pair[0] == pair_vals[1]: # TODO: this is not true. "st", "tr,ru,ue", -X-> "str,ru,ue"
                        new_entries.append(((pair_vals[0], merged_token), pretoken_counts[token_bytes]['pair_counts'][pair]))
                        del_pairs.append(pair)
                    elif merge_pair[1] == pair[0]:
                        new_entries.append(((merged_token, pair[1]), pretoken_counts[token_bytes]['pair_counts'][pair]))
                        del_pairs.append(pair)

            for entry, pair in zip(new_entries, del_pairs):
                pretoken_counts[token_bytes]['pair_counts'][entry[0]] = entry[1]
                del pretoken_counts[token_bytes]['pair_counts'][pair]

    return pretoken_counts


def apply_bpe(pretoken_counts, vocabulary, merges, vocab_size, idx):
    # Apply the BPE algorithm and train the tokenizer until the desired vocab size is reached
    while len(vocabulary) < vocab_size:
        # Determine overall pair counts
        total_pair_counts = get_total_pair_counts(pretoken_counts=pretoken_counts)

        # Calculate next merge
        merge_pair = max(get_dict_keys_with_max_vals(total_pair_counts))
        merges.append(merge_pair)

        # Add merge to vocab
        merged_token = merge_pair[0] + merge_pair[1]
        vocabulary[idx] = merged_token
        idx += 1

        # Propagate the merge to the pretoken counts
        pretoken_counts = propagate_new_merge(pretoken_counts=pretoken_counts, merge_pair=merge_pair, merged_token=merged_token)

    return vocabulary, merges


def train_bpe_tokenizer(input_path: str,
                        vocab_size: int,
                        special_tokens: List[str]) -> (Dict[int, bytes], List[Tuple[bytes, bytes]]):
    # Load text
    text = open(input_path).read()

    # Prepare vars
    merges: List[Tuple[bytes, bytes]] = []

    # Init vocab
    vocabulary, idx = initialize_vocab(special_tokens)

    # Pre-tokenize the text and get initial pretoken counts
    pretokenization_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenized_text = re.findall(pretokenization_regex, text)

    pretoken_counts: Dict[bytes, Dict] = {}
    for token in pretokenized_text:
        if token not in special_tokens:
            token_bytes = token.encode(ENCODING)

            if token_bytes in pretoken_counts:
                pretoken_counts[token_bytes]['count'] += 1
            else:
                pretoken_counts[token_bytes] = {}
                pretoken_counts[token_bytes]['count'] = 1
                pretoken_counts[token_bytes]['pair_counts'] = get_pair_counts(token_bytes)

    # Train the tokenizer via BPE algorithm
    vocabulary, merges = apply_bpe(pretoken_counts=pretoken_counts,
                                   vocabulary=vocabulary,
                                   merges=merges,
                                   vocab_size=vocab_size,
                                   idx=idx)

    ppr.pprint(vocabulary)
    ppr.pprint(merges)

    return vocabulary, merges


def train_bpe_tokenizer2(input_path: str,
                        vocab_size: int,
                        special_tokens: List[str]) -> (Dict[int, bytes], List[Tuple[bytes, bytes]]):
    # Load text
    text = open(input_path).read()

    # Prepare vars
    merges: List[Tuple[bytes, bytes]] = []

    # Init vocab
    vocabulary, idx = initialize_vocab(special_tokens)

    # Pre-tokenize the text
    pretokenization_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenized_text = re.findall(pretokenization_regex, text)

    # Get pretoken counts
    pretoken_counts = {}
    for elem in pretokenized_text:
        elem_bytes = elem.encode(ENCODING)
        if elem_bytes in pretoken_counts:
            pretoken_counts[elem_bytes] += 1
        else:
            pretoken_counts[elem_bytes] = 1

    # Get pair to pretoken mappings
    pairs_to_pretokens = {}
    for elem_bytes in pretoken_counts:
        for i in range(0, len(elem_bytes) - 1, 1):
            byte1 = bytes([elem_bytes[i]])
            byte2 = bytes([elem_bytes[i + 1]])
            pair = (byte1, byte2)

            if pair in pairs_to_pretokens:
                pairs_to_pretokens[pair].append(elem_bytes)
            else:
                pairs_to_pretokens[pair] = [elem_bytes]

    # Get number of occurrences of each pair
    pair_counts = {}
    for pair in pairs_to_pretokens:
        pair_counts[pair] = sum([pretoken_counts[elem_bytes] for elem_bytes in pairs_to_pretokens[pair]])

    # Find the pair to merge, then merge it
    merge_pair = max(get_dict_keys_with_max_vals(pair_counts))
    new_token = merge_pair[0] + merge_pair[1]
    vocabulary[idx] = new_token
    idx += 1

    # Update


def convert_to_bytes(input):
    output = []
    for elem in tuple(input.encode(ENCODING)):
        output.append(elem.to_bytes(1, 'big'))

    return tuple(output)


def apply_bpe_algo(pretoken_counts, vocabulary, vocab_size, merges, idx):
    while len(vocabulary) < vocab_size:
        # Get number of occurrences of each pair explicitly
        pair_counts = {}
        for elem_bytes in pretoken_counts:
            pretoken_count = pretoken_counts[elem_bytes]

            for i in range(0, len(elem_bytes) - 1, 1):
                byte1 = elem_bytes[i]
                byte2 = elem_bytes[i + 1]
                pair = (byte1, byte2)

                if pair in pair_counts:
                    pair_counts[pair] += 1 * pretoken_count
                else:
                    pair_counts[pair] = 1 * pretoken_count

        # ppr.pprint(pair_counts)

        # Find the pair to merge, then merge it
        merge_pair = max(get_dict_keys_with_max_vals(pair_counts))
        merges.append(merge_pair)
        new_token = merge_pair[0] + merge_pair[1]
        vocabulary[idx] = new_token
        idx += 1
        # print(merge_pair, new_token)

        # Update pretokens
        new_elems = []
        del_elems = []
        for elem_bytes in pretoken_counts:
            # Identify if the new token is in this pretoken
            to_search = b''.join(elem_bytes)
            if new_token in to_search:
                new_elem = []

                # Find all the places where it should be replaced
                skip = False
                for i in range(len(elem_bytes)):
                    if skip:
                        skip = False
                        continue

                    if elem_bytes[i] == merge_pair[0] and i < len(elem_bytes) - 1 and elem_bytes[i + 1] == merge_pair[1]:
                        new_elem.append(new_token)
                        skip = True
                    else:
                        new_elem.append(elem_bytes[i])

                new_elems.append((tuple(new_elem), pretoken_counts[elem_bytes]))
                del_elems.append(elem_bytes)

        # Update keys with new tokenization
        for new_elem in new_elems:
            pretoken_counts[new_elem[0]] = new_elem[1]
        for del_elem in del_elems:
            del pretoken_counts[del_elem]

    return vocabulary, merges


def train_bpe_tok(input_path: str,
                  vocab_size: int,
                  special_tokens: List[str]) -> (Dict[int, bytes], List[Tuple[bytes, bytes]]):
    # Load text
    text = open(input_path).read()

    # Prepare vars
    merges: List[Tuple[bytes, bytes]] = []

    # Init vocab
    vocabulary, idx = initialize_vocab(special_tokens)

    # Pre-tokenize the text
    pretokenization_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenized_text = re.findall(pretokenization_regex, text)

    # Get pretoken counts
    pretoken_counts = {}
    for elem in pretokenized_text:
        elem_bytes = convert_to_bytes(elem)
        # print(elem_bytes)

        if elem_bytes in pretoken_counts:
            pretoken_counts[elem_bytes] += 1
        else:
            pretoken_counts[elem_bytes] = 1

    # Train tokenizer up
    vocabulary, merges = apply_bpe_algo(pretoken_counts=pretoken_counts, vocabulary=vocabulary, vocab_size=vocab_size, merges=merges, idx=idx)

    # ppr.pprint(vocabulary)
    # ppr.pprint(merges)

    return vocabulary, merges


def main():
    train_bpe_tok(input_path='test.txt',
                        vocab_size=262,
                        special_tokens=['<|Special!|>'])


if __name__ == '__main__':
    main()
