import torch
from torch.utils.data import DataLoader
import os
import time

from model import Golu
from tokenizer import ByteLevelTokenizer

from config import Config
cfg = Config()

torch.manual_seed(42)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_device(device)
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'


tokenizer = ByteLevelTokenizer()
dataloader = DataLoader(tokenizer, batch_size=cfg.batch)

model = Golu()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
best_test_loss = 1_000_000.

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=0.6)

if os.path.exists('current_model.pt'):
    print('\nLoading existing model\n')
    checkpoint = torch.load('current_model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_test_loss = checkpoint['loss']

model.compile()
model.print_model_info()


for epoch in range(cfg.epochs):
    print(scheduler.get_last_lr())
    epoch_start_time = time.time()
    start_time = time.time()
    running_train_loss = 0.
    print(f"\nEPOCH {epoch+1}:")
    step = 0
    for input, output in dataloader:
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model.forward(input, output)
        loss.backward()
        optimizer.step()
        step += 1
        running_train_loss += loss.item()
        if step % 100 == 99:
            last_loss = running_train_loss/100
            elapsed = str(round(time.time() - start_time, 2)) + "s"
            print(f"step: {step+1:<10} time: {elapsed:<10} row: {tokenizer.row:<10}  train_loss: \033[1;92m{
                last_loss:<10.4f}\033[0m")
            running_train_loss = 0.
            start_time = time.time()

        if step % 1000 == 999:
            with torch.no_grad():
                prompt = "<USER> "
                model.eval()
                print("\n")
                model.generate(torch.tensor([tokenizer.encode(prompt)]))
                print("\n\n")
                model.train()

                print(f"Saving recent model with loss {last_loss}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': last_loss
                }, 'current_model.pt')

            if cfg.save_model and last_loss < best_test_loss:
                best_test_loss = last_loss
                print(f"Saving New best model with loss {best_test_loss}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': best_test_loss
                }, 'model.pt')
