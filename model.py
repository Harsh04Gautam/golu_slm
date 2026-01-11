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
        self.lang_mode_head = nn.Linear(
            cfg.num_embed, token.vocab * cfg.prediction)
        self.blocks = nn.Sequential(
            *[nn.Sequential(*[Block(min(cfg.kernel * (2**i), cfg.block), 1)]) for i in range(cfg.num_layer)])
        self.sa_block = Block(cfg.prediction, 1, True)
        self.layer_norm = nn.LayerNorm(cfg.num_embed)

    def forward(self, x, y=None):
        _, T = x.shape
        x = self.token_embedding(x)
        x = x + self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lang_mode_head(x)
        B, T, C = logits.shape
        logits = logits.view(B, T, cfg.prediction, token.vocab)
        logits = self.sa_block(logits)
        if y is None:
            return logits[:, :, 0, :], None
        logits = logits.view(B, T * cfg.prediction, token.vocab)
        y = y.unfold(1, cfg.prediction, 1)
        _, T, _ = y.shape
        logits = logits[:, :T*cfg.prediction]
        loss = F.cross_entropy(logits.reshape(
            B*T*cfg.prediction, token.vocab), y.reshape(B*T*cfg.prediction))
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
    def __init__(self, kernel, step=None, self_attention=False):
        super().__init__()
        step = kernel - 2 if step is None else step
        ln_size = cfg.num_embed if not self_attention else token.vocab
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(ln_size) for _ in range(2)])
        self.attention = MultiHeadPatchAttention(
            kernel, step, self_attention)
        self.feed_forward = FeedForward(self_attention)
        self.dropout = nn.ModuleList(
            [nn.Dropout(cfg.dropout) for _ in range(2)])

    def forward(self, x):
        x = x + self.dropout[0](self.attention(self.layer_norm[0](x)))
        x = x + self.dropout[1](self.feed_forward(self.layer_norm[1](x)))
        return x


class MultiHeadPatchAttention(nn.Module):
    def __init__(self, kernel, step, self_attention=False):
        super().__init__()
        self.self_attention = self_attention
        embed = cfg.num_embed if not self_attention else token.vocab
        self.kernel = kernel
        self.step = step
        self.qkv = nn.Linear(embed, 3*cfg.head, bias=False)
        self.add_rotation = Rotation()
        self.proj = nn.Linear(cfg.num_embed, embed)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer('mask', torch.tril(
            torch.ones((kernel, kernel))))
        self.register_buffer('mask_sliding', torch.tril(
            torch.ones((kernel, kernel)), kernel-step))
        self.indices = None

    def forward(self, x: torch.tensor):
        if self.self_attention:
            B, T, P, C = x.shape
            H = cfg.head//cfg.num_head

            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = self.add_rotation(
                q.view(B, T, P, cfg.num_head, H).transpose(2, 3))
            k = self.add_rotation(
                k.view(B, T, P, cfg.num_head, H).transpose(2, 3))
            v = v.view(B, T, P, cfg.num_head, H).transpose(2, 3)
            x = self._self_attention(q, k, v)
            x = x.transpose(2, 3).reshape(B, T, P, cfg.num_head * H)
            x = self.proj(x)
            return x

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

        x = x.transpose(1, 2).reshape(B, T, cfg.num_head * H)

        return x

    def _self_attention(self, q, k, v, mask=None):
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        if mask is not None:
            wei = wei.masked_fill(
                mask[:wei.shape[-2], :wei.shape[-1]] == 0, float('-inf'))
        wei = wei.softmax(-1)
        wei = self.dropout(wei)
        return wei @ v

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
    def __init__(self, self_attention=False):
        super().__init__()
        embed = cfg.num_embed if not self_attention else token.vocab
        self.feed_forward = nn.Sequential(
            nn.Linear(embed, embed * 4),
            nn.GELU(),
            nn.Linear(embed * 4, embed))

    def forward(self, x):
        return self.feed_forward(x)
