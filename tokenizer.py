import torch
from torch.utils.data import IterableDataset
import codecs
from datasets import load_dataset

from config import Config

cfg = Config()


class ByteLevelTokenizer(IterableDataset):
    def __init__(self):
        self.vocab = 256 + 3
        self.prompt = self.vocab - 3
        self.text = self.vocab - 2
        self.end = self.vocab - 1
        self.row = 0

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        tokens = bytes([x if x < 256 else 10 for x in tokens])  # "\n" -> 10
        return codecs.decode(tokens, encoding="utf-8", errors="replace")

    # def __iter__(self):
    #     fw = load_dataset("stingning/ultrachat",
    #                       name="default", split="train", streaming=True).shuffle()
    #     for data in fw:
    #         data = data['data']
    #         user = "<USER>"
    #         assistant = "<ASSISTANT>"
    #         speaker = user
    #         tokens = []
    #         for text in data:
    #             conv = self.encode(speaker + " " + text + "\n")
    #             tokens += conv + [self.end]
    #             speaker = user if speaker == assistant else assistant
    #
    #         for i in range(0, len(tokens) - cfg.block, cfg.block):
    #             chunk = tokens[i: i + cfg.block + 1]
    #
    #             if len(chunk) < cfg.block + 1:
    #                 continue
    #
    #             x = torch.tensor(chunk[:-1])
    #             y = torch.tensor(chunk[1:])
    #             yield x, y

    # def __iter__(self):
    #     fw = load_dataset("lmsys/lmsys-chat-1m",
    #                       name="default", split="train", streaming=True).shuffle()
    #     for data in fw:
    #         data = data["conversation"]
    #         tokens = [self.text] + self.encode("<USER>") + self.encode(data[0]['content']) + self.encode(
    #             "<ASSISTANT>") + self.encode(data[1]['content']) + [self.end]
    #         for i in range(0, len(tokens) - cfg.block, cfg.block):
    #             chunk = tokens[i: i + cfg.block + 1]
    #
    #             if len(chunk) < cfg.block + 1:
    #                 continue
    #
    #             x = torch.tensor(chunk[:-1])
    #             y = torch.tensor(chunk[1:])
    #             yield x, y

    def __iter__(self):
        fw = load_dataset("HuggingFaceFW/fineweb-edu",
                          name="sample-10BT", split="train", streaming=True).shuffle()
        for data in fw:
            self.row += 1
            tokens = [self.text] + self.encode(data['text']) + [self.end]
            for i in range(0, len(tokens) - cfg.block, cfg.block):
                chunk = tokens[i: i + cfg.block + 1]

                if len(chunk) < cfg.block + 1:
                    continue

                x = torch.tensor(chunk[:-1])
                y = torch.tensor(chunk[1:])
                yield x, y
