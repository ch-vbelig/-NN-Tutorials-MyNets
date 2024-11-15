import torch
import config
from tqdm import tqdm

def train_fn(model, data_loader, optimizer):
    # Set the model into train mode
    # Dropout and BatchNorm layers of the model -> ON
    model.run_training()

    fin_loss = 0.0
    tk = tqdm(data_loader, total=len(data_loader))

    for data in tk:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)

        # Forward pass
        _, loss = model(data["images"], data["targets"])
        fin_loss += loss.item()

        # Update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return fin_loss / len(data_loader)


def eval_fc(model, data_loader):
    # set the model into evaluation mode
    # Dropout and BatchNorm layers of the model -> OFF
    with torch.no_grad():

        model.eval()

        fin_loss = 0.0
        fin_preds = []
        tk = tqdm(data_loader, total=len(data_loader))

        for data in tk:
            for key, value in data.items():
                data[key] = value.to(config.DEVICE)

            # Forward pass
            batch_preds, loss = model(data["images"], data["targets"])
            fin_loss += loss.item()
            fin_preds.append(batch_preds)

        return fin_preds, fin_loss / len(data_loader)