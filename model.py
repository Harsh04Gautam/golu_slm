import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from tokenizer import ByteLevelTokenizer

token = ByteLevelTokenizer()
cfg = Config()


class Golu(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(token.vocab, cfg.num_embed)
        self.lang_mode_head = nn.Linear(cfg.num_embed, token.vocab)
        self.blocks = nn.Sequential(
            *[nn.Sequential(*[Block(min(cfg.kernel * (2**i), 128), 1)]) for i in range(cfg.num_layer)])
        self.layer_norm = nn.LayerNorm(cfg.num_embed)

    def forward(self, x, y=None):
        _, T = x.shape
        x = self.token_embedding(x)
        x = x + self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lang_mode_head(x)
        if y is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
        return logits, loss

    def print_model_info(self):
        print("\nMODEL INFO")
        print(f"{"total parameters":<20}: \033[1;92m{sum(p.numel()
              for p in self.parameters())}\033[0m")
        cfg.print_config()

    def generate(self, x):
        while True:
            x = x[:, -cfg.block:]
            logits, loss = self.forward(x, None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            vals, indices = torch.topk(probs, 256)
            index = indices[0][torch.multinomial(vals, num_samples=1)]
            if x.shape[1] > cfg.max_tokens:
                break
            x = torch.cat(
                (x, index), dim=1)
            print(f"{token.decode([x[0][-1]])}", end="", flush=True)
        return x[0]


class Block(nn.Module):
    def __init__(self, kernel, step=None):
        super().__init__()
        step = kernel - 2 if step is None else step
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(cfg.num_embed) for _ in range(2)])
        self.attention = MultiHeadPatchAttention(
            kernel, step)
        self.feed_forward = FeedForward()
        self.dropout = nn.ModuleList(
            [nn.Dropout(cfg.dropout) for _ in range(2)])

    def forward(self, x):
        x = x + self.dropout[0](self.attention(self.layer_norm[0](x)))
        x = x + self.dropout[1](self.feed_forward(self.layer_norm[1](x)))
        return x


class MultiHeadPatchAttention(nn.Module):
    def __init__(self, kernel, step):
        super().__init__()
        self.kernel = kernel
        self.step = step
        self.qkv = nn.Linear(cfg.num_embed, 3*cfg.head, bias=False)
        self.add_rotation = Rotation()
        # self.proj = nn.Linear(cfg.head, cfg.num_embed)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer('mask', torch.tril(
            torch.ones((kernel, kernel))))
        self.register_buffer('mask_sliding', torch.tril(
            torch.ones((kernel, kernel)), kernel-step))
        self.indices = None

    def forward(self, x: torch.tensor):
        S = self.kernel
        B, T, C = x.shape
        H = cfg.head//cfg.num_head

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.add_rotation(q.view(B, T, cfg.num_head, H).transpose(1, 2))
        k = self.add_rotation(k.view(B, T, cfg.num_head, H).transpose(1, 2))
        v = v.view(B, T, cfg.num_head, H).transpose(1, 2)

        x = self._attention(q[:, :, :S], k[:, :, :S], v[:, :, :S], self.mask)

        if T > S:
            k_new = k[:, :, 1:].unfold(2, S, self.step).transpose(-2, -1)
            v_new = v[:, :, 1:].unfold(2, S, self.step).transpose(-2, -1)
            q_new = self._unfold(q[:, :, S:], self.step)

            x_sliding = self._attention(q_new, k_new, v_new, self.mask_sliding)
            x_sliding = self._fold(x_sliding, T - S)

            x = torch.cat((x, x_sliding), dim=-2)

        # look_back = (S * cfg.num_layer)//4
        # n = (T-look_back)//look_back
        # if T > look_back:
        #     if self.indices is None or T > len(self.indices):
        #         self.indices = torch.arange(look_back, T, look_back)
        #     q = q[:, :, self.indices[:n]].unsqueeze(-2)
        #     k = k[:, :, :look_back*n].view(B, cfg.num_head, n, look_back, H)
        #     v = v[:, :, :look_back*n].view(B, cfg.num_head, n, look_back, H)
        #     x[:, :, self.indices[:n]] = self._attention(q, k, v).squeeze(-2)

        x = x.transpose(1, 2).reshape(B, T, cfg.num_head * H)
        # x = self.proj(x)

        return x

    def _attention(self, q, k, v, mask=None):
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        if mask is not None:
            wei = wei.masked_fill(
                mask[:wei.shape[-2], :wei.shape[-1]] == 0, float('-inf'))
        wei = wei.softmax(-1)
        wei = self.dropout(wei)
        return wei @ v

    def _expand(self, x, P):
        remainder = x.shape[-2] % P
        if remainder != 0:
            repeat = x[..., [-1], :].expand(x.shape[:-2] + (P-remainder, -1))
            x = torch.cat((x, repeat), dim=-2)

        return x

    def _unfold(self, x, P):
        x = self._expand(x, P)
        T, C = x.shape[-2:]
        return x.view(x.shape[:-2] + (math.ceil(T/P), P, C))

    def _fold(self, x, batch):
        T, P, C = x.shape[-3:]
        return x.reshape(x.shape[:-3] + (T * P, C))[..., :batch, :]


class Rotation(nn.Module):
    def __init__(self, base=10000):
        super().__init__()
        head = cfg.head//cfg.num_head
        self.inv_freq = 1.0 / base ** (torch.arange(0, head, 2) / head)
        self.register_buffer('inf_freq', self.inv_freq)
        self.freq = torch.outer(torch.arange(cfg.block).float(), self.inv_freq)
        self.cos = self.freq.cos()
        self.sin = self.freq.sin()

    def forward(self, x):
        if (x.shape[-2] > self.freq.shape[0]):
            self.freq = torch.outer(torch.arange(
                x.shape[-2]).float(), self.inv_freq)
            self.cos = self.freq.cos()
            self.sin = self.freq.sin()

        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([
            x1*self.cos[:x.shape[-2]] - x2*self.sin[:x.shape[-2]],
            x1*self.sin[:x.shape[-2]] + x2*self.cos[:x.shape[-2]]
        ], dim=-1)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(cfg.num_embed, cfg.num_embed * 4),
            nn.GELU(),
            nn.Linear(cfg.num_embed * 4, cfg.num_embed))

    def forward(self, x):
        return self.feed_forward(x)
