import torch
import os

from model import Golu
from tokenizer import ByteLevelTokenizer

from config import Config
cfg = Config()

# torch.manual_seed(42)

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'


tokenizer = ByteLevelTokenizer()

model = Golu()

if os.path.exists('current_model.pt'):
    print('Loading existing model')
    checkpoint = torch.load('current_model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

model.compile()
print("MODEL INFO")
print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n")

tokenizer.decode([tokenizer.prompt])

with torch.no_grad():
    model.eval()
    prompt = "\n"
    for _ in range(2):
        print("\n")
        print(prompt, end="")
        model.generate(torch.tensor([tokenizer.encode(prompt)]))
        print("\n\n")
