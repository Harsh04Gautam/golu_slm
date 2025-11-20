import torch
import os
import time

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


@torch.no_grad
def evaluate(step, eval_iteration, start_time):
    model.eval()
    train_loss = 0.0
    val_loss = 0.0
    elapsed = str(round(time.time() - start_time, 2)) + "s"

    for _ in range(eval_iteration):
        input, output = tokenizer.get_batch('train')
        _, loss = model.forward(input, output)
        train_loss = train_loss + loss

        input, output = tokenizer.get_batch('val')
        _, loss = model.forward(input, output)
        val_loss = val_loss + loss

    print(f"step: {step:<10} time: {elapsed:<10} train_loss: \033[1;92m{
          (train_loss/eval_iteration).item():<10.4f}\033[0m  val_loss: \033[1;92m{
          (val_loss/eval_iteration).item():<10.4f}\033[0m")

    if step % cfg.generate_interval == 0 and step != 0:
        print("\n")
        model.generate(torch.tensor([[0]]), cfg.max_tokens)
        print("\n\n")
    model.train()
    return time.time()


tokenizer = ByteLevelTokenizer(text)

model = Golu()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

if os.path.exists('model.pt'):
    print('\nLoading existing model\n')
    checkpoint = torch.load('model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.compile()
print("\nMODEL INFO")
print(f"{"total parameters":<20}: \033[1;92m{sum(p.numel()
      for p in model.parameters())}\033[0m")
for name, value in Config.__dict__.items():
    if not name.startswith("__"):
        print(f"{name:<20}: \033[1;92m{value}\033[0m")
print("\n")


start_time = time.time()
for step in range(cfg.total_iteration):
    optimizer.zero_grad()
    input, output = tokenizer.get_batch('train')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model.forward(input, output)
    loss.backward()
    optimizer.step()

    if step % cfg.eval_interval == 0:
        start_time = evaluate(step, cfg.eval_iteration, start_time)

if cfg.save_model:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'model.pt')
