import torch
import os

from model import Golu
from tokenizer import ByteLevelTokenizer

from config import Config
cfg = Config()

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'

# data loader
with open("input.txt") as f:
    text = f.read()


tokenizer = ByteLevelTokenizer(text)

model = Golu()

if os.path.exists('model.pt'):
    print('Loading existing model')
    checkpoint = torch.load('model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

model.compile()
print("MODEL INFO")
print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n")


with torch.no_grad():
    model.eval()
    print("\n")
    model.generate(torch.tensor([[0]]), cfg.max_tokens)
    print("\n\n")
