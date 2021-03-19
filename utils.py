import torch

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

