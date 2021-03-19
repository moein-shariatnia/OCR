import torch

PATH = "./input/captcha_images_v2"
BATCH_SIZE = 32
WIDTH = 300
HEIGHT = 75
EPOCHS = 50
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
DROPOUT = 0.2
PATIENCE = 3
FACTOR = 0.8
