import torch
import torch.nn as nn
import numpy as np
import math
import pprint
import collections
from collections import OrderedDict
from typing import Optional
import os
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import json
from cs336_basics import tokenizer_class_2
import random
from torch.nn.init import xavier_uniform_, normal_

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"
ENCODING = 'utf-8'
ppr = pprint.PrettyPrinter()


class RMSNorm(nn.Module):
    def __init__(self, d_model, weights=None, eps=1e-5):
        super().__init__()

        self.d_model = d_model

        if weights is not None:
            print(type(weights))
            if type(weights) == dict or type(weights) == OrderedDict:
                self.weights = nn.Parameter(data=weights['weight'])
            else:
                self.weights = nn.Parameter(data=weights)
        else:
            self.weights = nn.Parameter(data=torch.rand(d_model))

        self.eps = eps

    def forward(self, in_features):
        sum_square = torch.square(in_features)
        sum_square = torch.sum(sum_square, dim=-1, keepdim=True)
        rms_denominator = torch.sqrt((1 / self.d_model) * sum_square + self.eps)

        return (in_features / rms_denominator) * self.weights


class FeedForward(nn.Module):
    def __init__(self, d_model=None, d_ff=None, weights=None):
        super().__init__()

        self.d_model = d_model
        if d_ff:
            self.d_ff = d_ff
        elif self.d_model:
            self.d_ff = 4 * self.d_model

        if weights:
            if 'w1.weight' in weights:
                self.w1 = nn.Parameter(data=weights['w1.weight'])
                self.w2 = nn.Parameter(data=weights['w2.weight'])
        elif self.d_model:
            self.w1 = nn.Parameter(data=xavier_uniform_(torch.empty([self.d_ff, self.d_model])))
            self.w2 = nn.Parameter(data=xavier_uniform_(torch.empty([self.d_model, self.d_ff])))
        else:
            self.w1 = None
            self.w2 = None

        # self.to(torch.device('cuda')) # TODO: bandaid

    def apply_gelu(self, in_features):
        return in_features * 0.5 * (1 + torch.erf(in_features / math.sqrt(2)))

    def forward(self, in_features):
        return self.apply_gelu(in_features @ self.w1.T) @ self.w2.T


class ScaledDotProdAttn(nn.Module):
    def __init__(self):
        super().__init__()
        # self.to(torch.device('cuda'))  # TODO: bandaid

    def apply_softmax(self, in_features, dim):
        conditioned_features = in_features - torch.max(in_features, dim=dim, keepdim=True).values
        return torch.exp(conditioned_features) / torch.sum(torch.exp(conditioned_features), dim=dim, keepdim=True)

    def forward(self, K, Q, V, mask=None, pdrop=None):
        pre_smax = (Q @ K.transpose(-1, -2)) / math.sqrt(K.shape[-1])

        if mask is not None:
            pre_smax.masked_fill_(mask=mask, value=float('-inf'))

        post_smax = self.apply_softmax(pre_smax, dim=-1)
        if pdrop:
            post_smax = nn.functional.dropout(input=post_smax, p=pdrop)

        return post_smax @ V


class CausalMultiHeadSelfAttn(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop=None, weights=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        self.d_qkv = int(d_model / num_heads)

        # print(d_model, num_heads, self.d_qkv)

        self.attn_p_drop = attn_pdrop

        self.q_heads = xavier_uniform_(torch.empty([num_heads, self.d_qkv, d_model]))
        self.k_heads = xavier_uniform_(torch.empty([num_heads, self.d_qkv, d_model]))
        self.v_heads = xavier_uniform_(torch.empty([num_heads, self.d_qkv, d_model]))
        self.output_proj = xavier_uniform_(torch.empty([d_model, d_model]))

        if weights is not None:
            if len(weights) == 3 * num_heads + 1:
                for i in range(num_heads):
                    self.q_heads[i] = weights[f'q_heads.{i}.weight']
                    self.k_heads[i] = weights[f'k_heads.{i}.weight']
                    self.v_heads[i] = weights[f'v_heads.{i}.weight']
            elif len(weights) == 3 + 1:
                self.q_heads = torch.stack(torch.chunk(weights['q_proj.weight'], chunks=num_heads, dim=-2), dim=0)
                self.k_heads = torch.stack(torch.chunk(weights['k_proj.weight'], chunks=num_heads, dim=-2), dim=0)
                self.v_heads = torch.stack(torch.chunk(weights['v_proj.weight'], chunks=num_heads, dim=-2), dim=0)
                assert list(self.q_heads.shape) == [num_heads, self.d_qkv, d_model]

            self.output_proj = weights['output_proj.weight']

        self.q_heads = nn.Parameter(self.q_heads)
        self.k_heads = nn.Parameter(self.k_heads)
        self.v_heads = nn.Parameter(self.v_heads)
        self.output_proj = nn.Parameter(self.output_proj)

        # self.to(torch.device('cuda'))  # TODO: bandaid

    def apply_softmax(self, in_features, dim):
        conditioned_features = in_features - torch.max(in_features, dim=dim, keepdim=True).values
        return torch.exp(conditioned_features) / torch.sum(torch.exp(conditioned_features), dim=dim, keepdim=True)

    def apply_attention(self, K, Q, V, mask=None, pdrop=None):
        pre_smax = (Q @ K.transpose(-1, -2)) / math.sqrt(K.shape[-1])

        if mask is not None:
            pre_smax.masked_fill_(mask=mask, value=float('-inf'))

        post_smax = self.apply_softmax(pre_smax, dim=-1)

        if pdrop:
            post_smax = nn.functional.dropout(input=post_smax, p=pdrop)

        return post_smax @ V

    def forward(self, in_features):
        # Get inputs to attention
        Qq = in_features.unsqueeze(dim=1) @ self.q_heads.unsqueeze(dim=0).transpose(-1, -2)
        Kk = in_features.unsqueeze(dim=1) @ self.k_heads.unsqueeze(dim=0).transpose(-1, -2)
        Vv = in_features.unsqueeze(dim=1) @ self.v_heads.unsqueeze(dim=0).transpose(-1, -2)

        # Run attention
        seq_len = in_features.shape[-2]
        mask = torch.triu(torch.ones([seq_len, seq_len]).to(dtype=torch.bool), diagonal=1)
        mask = mask.to(in_features.device)
        attn_output = self.apply_attention(Kk, Qq, Vv, mask=mask, pdrop=self.attn_p_drop)

        # Concat each head's output and project final values
        output = (torch.cat(attn_output.chunk(chunks=self.num_heads, dim=1), dim=-1) @ self.output_proj.T).squeeze(dim=1)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, weights=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.weights = weights

        self.norm1 = RMSNorm(d_model=self.d_model,
                             weights=None if weights is None else weights['ln1.weight'])
        self.attention = CausalMultiHeadSelfAttn(d_model=self.d_model,
                                                 num_heads=self.num_heads,
                                                 attn_pdrop=self.attn_pdrop,
                                                 weights=None if weights is None else {'output_proj.weight': weights['attn.output_proj.weight'],
                                                                                       'q_proj.weight': weights['attn.q_proj.weight'],
                                                                                       'k_proj.weight': weights['attn.k_proj.weight'],
                                                                                       'v_proj.weight': weights['attn.v_proj.weight']})

        self.norm2 = RMSNorm(d_model=self.d_model,
                             weights=None if weights is None else weights['ln2.weight'])
        self.ffnet = FeedForward(d_model=self.d_model,
                                 d_ff=self.d_ff,
                                 weights=None if weights is None else {'w1.weight': weights['ffn.w1.weight'],
                                                                       'w2.weight': weights['ffn.w2.weight']})
        self.residual_dropout = nn.Dropout(p=residual_pdrop)

        # self.to(torch.device('cuda')) # TODO: bandaid

    def forward(self, in_features):
        sublayer_output1 = in_features + self.residual_dropout(self.attention(self.norm1(in_features)))
        sublayer_output2 = sublayer_output1 + self.residual_dropout(self.ffnet(self.norm2(sublayer_output1)))

        return sublayer_output2


class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 attn_pdrop,
                 residual_pdrop,
                 vocab_size,
                 context_length,
                 num_layers,
                 tokenizer_vocab_fp=None,
                 tokenizer_merges_fp=None,
                 special_tokens=None,
                 weights=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.weights = weights
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.token_embeddings = torch.rand([self.vocab_size, self.d_model]) * 2 * math.sqrt(1 / self.vocab_size) - math.sqrt(1 / self.vocab_size)
        self.positional_embeddings = normal_(torch.empty([self.context_length, self.d_model]))
        # self.token_embeddings = nn.Linear(self.vocab_size, self.d_model, bias=False)
        # self.positional_embeddings = nn.Embedding(num_embeddings=self.context_length,
        #                                           embedding_dim=self.d_model)
        if weights is not None:
            self.token_embeddings = weights['token_embeddings.weight']
            self.positional_embeddings = weights['position_embeddings.weight']

        self.token_embeddings = nn.Parameter(self.token_embeddings)
        self.positional_embeddings = nn.Parameter(self.positional_embeddings)

        self.residual_dropout = nn.Dropout(p=residual_pdrop)

        self.block_layers = {}
        for i in range(self.num_layers):
            if weights is not None:
                self.block_layers[i] = TransformerBlock(d_model=d_model,
                                                        num_heads=num_heads,
                                                        d_ff=d_ff,
                                                        attn_pdrop=attn_pdrop,
                                                        residual_pdrop=residual_pdrop,
                                                        weights={'ln1.weight': weights[f'layers.{i}.ln1.weight'],
                                                                 'ln2.weight': weights[f'layers.{i}.ln2.weight'],
                                                                 'attn.output_proj.weight': weights[f'layers.{i}.attn.output_proj.weight'],
                                                                 'attn.q_proj.weight': weights[f'layers.{i}.attn.q_proj.weight'],
                                                                 'attn.k_proj.weight': weights[f'layers.{i}.attn.k_proj.weight'],
                                                                 'attn.v_proj.weight': weights[f'layers.{i}.attn.v_proj.weight'],
                                                                 'ffn.w1.weight': weights[f'layers.{i}.ffn.w1.weight'],
                                                                 'ffn.w2.weight': weights[f'layers.{i}.ffn.w2.weight']})
            else:
                self.block_layers[i] = TransformerBlock(d_model=d_model,
                                                        num_heads=num_heads,
                                                        d_ff=d_ff,
                                                        attn_pdrop=attn_pdrop,
                                                        residual_pdrop=residual_pdrop,
                                                        weights=None)

            self.add_module(name=f'transformer-block-layer-{i}', module=self.block_layers[i])

        self.final_norm = RMSNorm(d_model=self.d_model,
                                  weights=None if weights is None else weights['ln_final.weight'])

        self.tokenizer_vocab_fp = tokenizer_vocab_fp
        self.tokenizer_merges_fp = tokenizer_merges_fp
        self.special_tokens = special_tokens

    def apply_softmax(self, in_features, dim=-1):
        conditioned_features = in_features - torch.max(in_features, dim=dim, keepdim=True).values
        return torch.exp(conditioned_features) / torch.sum(torch.exp(conditioned_features), dim=dim, keepdim=True)

    def get_tokenizer(
            self,
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
            self,
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
        return self.get_tokenizer(vocab, merges, special_tokens)

    # TODO: not sure if index selection is ok for backprop, since we want to train the token embedding weights
    def forward(self, in_indices) -> torch.Tensor:
        token_embeds = self.token_embeddings[in_indices]
        position_embds = self.positional_embeddings[
            [
                [
                    [i for i in range(in_indices.shape[1])]
                ] * in_indices.shape[0]
            ]
        ]
        # token_embeds = self.token_embeddings[in_indices]
        # pos_idxs = torch.ones([in_indices.shape[0], in_indices.shape[1]])
        # position_embds = self.positional_embeddings.forward(pos_idxs)

        layer_info = self.residual_dropout(token_embeds + position_embds)
        for block in self.block_layers:
            layer_info = self.block_layers[block](in_features=layer_info)

        logits = self.final_norm(in_features=layer_info) @ self.token_embeddings.transpose(-1, -2)

        return logits

    def apply_temperature_scaled_softmax(self, in_features, temperature, dim=-1):
        conditioned_features = in_features - torch.max(in_features, dim=dim, keepdim=True).values
        return torch.exp(conditioned_features / temperature) / torch.sum(torch.exp(conditioned_features / temperature), dim=dim, keepdim=True)

    def sample_token_from_dist(self, token_dist, threshold):
        token_dist_sorted = [[i, prob] for i, prob in zip(token_dist, range(len(token_dist)))]
        token_dist_sorted.sort(key=lambda item: item[1])

        sample_dist = []
        while threshold > 0 and len(sample_dist) < token_dist.shape[0]:
            new_elem = token_dist_sorted.pop()
            sample_dist.append(new_elem)
            threshold -= new_elem[0]
        # print(len(sample_dist))
        return random.choices([elem[1] for elem in sample_dist], weights=[elem[0] for elem in sample_dist])[0]

    # TODO fix batch sampling
    def decode(self, prompt, max_tokens=None, temperature=1, top_p=0.2):
        if type(prompt) is not str:
            raise NotImplementedError('Haven\'t configured to work with batch prompts yet')

        tokenizer = self.get_tokenizer_from_vocab_merges_path(vocab_path=self.tokenizer_vocab_fp,
                                                              merges_path=self.tokenizer_merges_fp,
                                                              special_tokens=self.special_tokens)
        token_idxs = tokenizer.encode(text=prompt)
        # print('id len', len(token_idxs))

        output = []
        next_token = None
        num_generated = 0
        while next_token != 0 and (max_tokens is None or num_generated < max_tokens):
            # print('NUM GEN', num_generated)
            logits = self.forward(in_indices=np.array([token_idxs]))
            next_token_dist = self.apply_temperature_scaled_softmax(in_features=logits[0,-1],
                                                                    temperature=temperature)

            next_token = self.sample_token_from_dist(next_token_dist, threshold=top_p)
            output.append(next_token)

            token_idxs.append(next_token)
            if len(token_idxs) > self.context_length:
                token_idxs = token_idxs[1:]

            num_generated += 1

        print('PROMPT:', prompt + '...')
        print('GENERATION:...' + tokenizer.decode(output))
        return tokenizer.decode(output)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        conditioned_logits = logits - torch.max(logits, dim=-1, keepdim=True).values
        smaxed_and_logged = -(conditioned_logits - torch.log(torch.sum(torch.exp(conditioned_logits), dim=-1, keepdim=True))).gather(dim=-1, index=targets.unsqueeze(dim=-1))

        return torch.mean(smaxed_and_logged)


def main():
    print()


if __name__ == '__main__':
    main()


