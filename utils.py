import torch
import numpy as np

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.avg, self.sum, self.count = [0]*3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    
    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def decode_predictions(preds, encoder):
    """
    preds shape: (timestep, batch_size, num_classes + 1)
    """
    preds = preds.permute(1, 0, 2) # shape: (batch_size, timestep, num_classes + 1)
    preds = preds.argmax(dim=2).cpu().numpy() # shape: (batch_size, timestep)

    captcha_preds = []
    for pred in preds:
        temp = []
        for step in pred:
            if step == 0:
                temp.append("%") # representative of UNK tokens
            else:
                step -= 1
                token = encoder.inverse_transform([step])[0]
                temp.append(token)
        
        temp = "".join(temp)
        captcha_preds.append(temp)
    
    return captcha_preds

def process_preds(captcha_preds):
    offsets = []
    for pred in captcha_preds:
        in_unk = False
        temp = []
        for i, char in enumerate(pred):
            if not in_unk and char == "%":
                in_unk = True
                temp.append(i)
            elif in_unk and char == "%":
                continue
            elif in_unk and char != "%":
                in_unk = False
                temp.append(i)
        offsets.append(temp)
    
    offsets = np.array([offset[1:] for offset in offsets])
    offsets = offsets.reshape(-1, 5, 2)
    captcha_texts = []
    for offset, pred in zip(offsets, captcha_preds):
        temp = []
        for (start, end) in offset:
            text = pred[start: end]
            text_len = len(text)
            if text_len > 1 and all(char == text[0] for char in text) :
                text = text[0]
            temp.append(text)
        captcha_texts.append(temp)
    captcha_texts = ["".join(text) for text in captcha_texts]
    return captcha_texts

