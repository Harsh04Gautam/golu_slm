import torch

# from config import block_size, batch_size
from config import Config

cfg = Config()


class ByteLevelTokenizer:
    def __init__(self, text):
        self.vocab = 256

        self.train_split = self.encode(text[:int(0.8 * len(text))])
        self.val_split = self.encode(text[int(0.8 * len(text)):])

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def get_batch(self, split, block=cfg.block):
        encoding = self.train_split if split == "train" else self.val_split

        indices = torch.randint(
            0, len(encoding) - block, (cfg.batch,))
        input = torch.tensor([encoding[i:i+block]
                             for i in indices], dtype=torch.long)
        output = torch.tensor(
            [encoding[i+1:i+1+block] for i in indices],  dtype=torch.long)

        return input, output
