import os
import pathlib
import torch
import numpy as np
from sklearn import preprocessing, model_selection, metrics

import config
import datasetGRAY
import engine
import utils
from model import CaptchaModel


def run_training(image_files):
    targets_orig = [str(path).split('/')[-1][:-4] for path in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flattened = [c for clist in targets for c in clist]


    # print(targets_orig) # -> ['226md', '22d5n' ... ]
    # print(targets) # -> [['2', '2', '6', 'm', 'd'], ['2', '2', 'd', '5', 'n'], ...]
    # print(targets_flattened) # -> ['2', '2', '6', 'm', 'd', '2', '2', 'd', '5', 'n', '2', '3', '5', '6', ...]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flattened)
    targets_enc = [lbl_enc.transform(c) for c in targets]
    # print(targets_enc)  # -> [array([ 0,  0,  4, 13,  9]), array([ 0,  0,  9,  3, 14]), ...]

    targets_enc = np.array(targets_enc)
    targets_enc = np.array(targets_enc) + 1  # -> reserving the value of 0 for '~' character

    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        _,
        test_target_orig
    ) = model_selection.train_test_split(image_files, targets_enc, targets_orig, test_size=0.2, random_state=12)

    train_dataset = datasetGRAY.ClassificationDatasetGRAY(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    test_dataset = datasetGRAY.ClassificationDatasetGRAY(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )

    data = next(iter(train_loader))
    print(data)
    print(data['images'].size())


    # define model
    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)

    # optimizer
    learning_rate = config.LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, verbose=True
    )

    # train
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)

        valid_preds, test_loss = engine.eval_fc(model, test_loader)
        valid_captcha_preds = []

        for vp in valid_preds:
            current_preds = utils.decode_predictions(vp, lbl_enc)
            valid_captcha_preds.extend(current_preds)

        combined = list(zip(test_target_orig, valid_captcha_preds))

        print(combined[:10])
        test_dup_rem = [utils.remove_duplicates(c) for c in test_target_orig]


        accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)

        print(f"Epoch: {epoch}, Train loss: {train_loss}, Test Loss: {test_loss}, Accuracy: {accuracy}")

        scheduler.step(test_loss)



    # Save model
    ##########load checkpoint#####################

    checkpoint = {
        "epoch": config.EPOCHS,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }

    torch.save(checkpoint, config.CHECKPOINT_FILE)


if __name__ == "__main__":
    path = pathlib.Path(config.DATA_DIR)
    image_files = list(path.glob('*.png'))
    # image_files = ['captcha_images/2en7g.png', 'captcha_images/2g783.png', 'captcha_images/2n73f.png']
    # print(image_files)
    run_training(image_files)
