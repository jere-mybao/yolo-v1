import torch
from dataset.dataset import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()
print(f"Using {gpu_count} GPUs")
if gpu_count > 1:
    batch_size = 4 * gpu_count
    nworkers = 4 * gpu_count
else:
    batch_size = 4
    nworkers = 2

weight_decay = 0.0005
epochs = 200


def train(model, train_loader, optimizer, loss_fn):
    accumulation_steps = 16 
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        with torch.set_grad_enabled(True):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    return float(loss.item())


def test(model, test_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
    return float(loss.item())

def main():
    pass  