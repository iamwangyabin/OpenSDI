import torch
from torch import nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from pkg_resources import packaging
from typing import Union, List
from copy import deepcopy

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


class PromptLearner(nn.Module):
    def __init__(self, ctx_dim, language_length, language_depth, dtype):
        super().__init__()
        self.n_ctx = language_length
        self.text_encoder_n_ctx = language_depth
        self.ctx_pos = nn.Parameter(torch.empty(1, self.n_ctx, ctx_dim))
        self.ctx_neg = nn.Parameter(torch.empty(1, self.n_ctx, ctx_dim))
        nn.init.normal_(self.ctx_pos, std=0.02)
        nn.init.normal_(self.ctx_neg, std=0.02)
        self.prompt_prefix = " ".join(["X"] * self.n_ctx)
        self.dtype = dtype

    def forward(self, clip_model, text_input, device):
        prompts = [self.prompt_prefix + " " + text + "." for text in text_input]
        tokenized_prompts = []
        for p in prompts:
            tokenized_prompts.append(tokenize(p))
        tokenized_prompts = torch.cat(tokenized_prompts).to(device)

        with torch.no_grad():
            embedding_text = clip_model.token_embedding(tokenized_prompts).to(self.dtype)

        token_prefix = embedding_text[:, :1, :]
        token_suffix = embedding_text[:, 1 + self.n_ctx:, :]

        prompts_pos = torch.cat([token_prefix, self.ctx_pos.expand(len(text_input), -1, -1), token_suffix], dim=1)
        prompts_neg = torch.cat([token_prefix, self.ctx_neg.expand(len(text_input), -1, -1), token_suffix], dim=1)

        prompts = torch.cat([prompts_neg, prompts_pos], dim=0)
        tokenized_prompts = torch.cat([tokenized_prompts, tokenized_prompts], dim=0)
        return prompts, tokenized_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.to(clip_model.dtype)
        self.positional_embedding = clip_model.positional_embedding.to(clip_model.dtype)
        self.ln_final = clip_model.ln_final.to(clip_model.dtype)
        self.text_projection = clip_model.text_projection.to(clip_model.dtype)


    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x,_,_ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x









