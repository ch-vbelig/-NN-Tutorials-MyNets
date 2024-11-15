import dataset
import config
import torch
from model import BuryatLanguageModel
import engine


def run_training(train_dict_path, val_dict_path, model_save_path):
    train_dataset = dataset.LanguageDataset(train_dict_path)
    test_dataset = dataset.LanguageDataset(val_dict_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    # print(len(train_dataset))
    # data = next(iter(train_loader))
    # print(data['data'].size())

    # get encoder and decoder
    char_to_idx = train_dataset.get_encoder()
    idx_to_char = train_dataset.get_decoder()

    vocab_size = len(char_to_idx)

    # init model
    model = BuryatLanguageModel(vocab_size)
    model.to(config.DEVICE)

    # configure optimizer and loss function
    learning_rate = config.INITIAL_LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer, criterion)

        if (epoch + 1) % 20 == 0:
            val_loss = engine.eval_fn(model, test_loader, criterion)
            print(f'Train loss: {train_loss}, Val loss: {val_loss}')

    engine.save_model(model, model_save_path)


if __name__ == '__main__':
    run_training(
        config.TRAIN_DICT_PATH,
        config.VAL_DICT_PATH,
        config.MODEL_SAVE_PATH
    )
