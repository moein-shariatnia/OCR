import torch
import albumentations as A
import numpy as np

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets, size, resize=True):
        """
        size: tuple (h, w)
        """

        self.image_paths = image_paths
        self.targets = targets
        self.size = size
        self.resize = resize
        self.augmentations = A.Normalize(max_pixel_value=255.0, always_apply=True)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.resize:
            image = image.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)

        target = self.targets[idx]
        image = np.array(image)
        image = self.augmentations(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.long),
        }

    def __len__(self):
        return len(self.image_paths)
