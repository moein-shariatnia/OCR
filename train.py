import os
import glob
import torch
import numpy as np
from pprint import pprint

from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection

import config
from dataset import OCRDataset
from model import OCRModel
import engine
from utils import decode_predictions


def make_loader(mode="train", *args, **kwargs):
    dataset = OCRDataset(*args, **kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def main():
    image_paths = glob.glob(os.path.join(config.PATH, "*.png"))
    image_paths = [path.replace("\\", "/") for path in image_paths]
    targets = [path.split("/")[-1][:-4] for path in image_paths]
    targets_listed = [[char for char in target] for target in targets]
    targets_flattened = [char for target in targets_listed for char in target]

    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(targets_flattened)
    targets_encoded = np.array(
        [label_enc.transform(target) for target in targets_listed]
    )
    targets_encoded += 1  # to keep the "0" class for UNK chars

    (
        train_imgs,
        valid_imgs,
        train_enc_targets,
        valid_enc_targets,
        _,
        valid_targets,
    ) = model_selection.train_test_split(
        image_paths, targets_encoded, targets, test_size=0.1, random_state=42
    )

    train_loader = make_loader(
        mode="train",
        image_paths=train_imgs,
        targets=train_enc_targets,
        size=(config.HEIGHT, config.WIDTH),
        resize=True,
    )

    valid_loader = make_loader(
        mode="valid",
        image_paths=valid_imgs,
        targets=valid_enc_targets,
        size=(config.HEIGHT, config.WIDTH),
        resize=True,
    )

    model = OCRModel(num_classes=len(label_enc.classes_), dropout=config.DROPOUT).to(
        config.DEVICE
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.PATIENCE,
        factor=config.FACTOR,
        verbose=True,
    )

    if config.MODE == "train":
        best_loss = float("inf")
        for epoch in range(config.EPOCHS):
            model.train()
            _ = engine.train(model, train_loader, optimizer)
            
            model.eval()
            with torch.no_grad():
                valid_preds, valid_loss = engine.eval(model, valid_loader)

            captcha_preds = []
            for preds in valid_preds:
                preds_ = decode_predictions(preds, label_enc)
                captcha_preds.extend(preds_)
            
            print(f"Epoch: {epoch}")
            pprint(list(zip(valid_targets, captcha_preds))[:10])

            lr_scheduler.step(valid_loss.avg)
            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(model.state_dict(), "model.pt")
    else:
        model.load_state_dict(torch.load("./models/model.pt", map_location=config.DEVICE))
        model.eval()
        with torch.no_grad():
            valid_preds, valid_loss = engine.eval(model, valid_loader)
        captcha_preds = []
        for preds in valid_preds:
            preds_ = decode_predictions(preds, label_enc)
            captcha_preds.extend(preds_)
        
        pprint(list(zip(valid_targets, captcha_preds))[:10])
        return valid_loader, captcha_preds, valid_targets


if __name__ == "__main__":
    main()
