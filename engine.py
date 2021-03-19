import torch
from tqdm import tqdm
import config

from utils import AvgMeter

def train(model, loader, optimizer):
    tqdm_object = tqdm(loader, total=len(loader))
    loss_meter = AvgMeter()

    for batch in tqdm_object:
        batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        preds, loss = model(images=batch['images'], targets=batch['targets'])
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), count=batch['images'].size(0))
        tqdm_object.set_postfix(loss=loss_meter.avg)
    
    return loss_meter

def eval(model, loader):
    tqdm_object = tqdm(loader, total=len(loader))
    loss_meter = AvgMeter()
    all_preds = []
    for batch in tqdm_object:
        batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
        preds, loss = model(images=batch['images'], targets=batch['targets'])
        all_preds.append(preds)
        loss_meter.update(loss.item(), count=batch['images'].size(0))
        tqdm_object.set_postfix(loss=loss_meter.avg)
    
    return all_preds, loss_meter

