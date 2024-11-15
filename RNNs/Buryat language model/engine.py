import torch
import config
from tqdm import tqdm
from model import BuryatLanguageModel
from torch.utils.data import DataLoader


def train_fn(model: BuryatLanguageModel, data_loader: DataLoader, optimizer, loss_fn):
    model.train()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))

    for batch_data in tk0:
        for key, value in batch_data.items():
            batch_data[key] = value.to(config.DEVICE)

        output = model(batch_data['data'])
        loss = loss_fn(output, batch_data['targets'])

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        fin_loss += loss.item()

    return fin_loss / len(data_loader)


def eval_fn(model: BuryatLanguageModel, data_loader: DataLoader, loss_fn):
    model.eval()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        for batch_data in tk0:
            for key, value in batch_data.items():
                batch_data[key] = value.to(config.DEVICE)

            output = model(batch_data['data'])
            loss = loss_fn(output, batch_data['targets'])

            fin_loss += loss.item()

    return fin_loss / len(data_loader)


def save_model(model, path):
    torch.save(model.state_dict(), path)
