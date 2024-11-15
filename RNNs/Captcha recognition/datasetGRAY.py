import albumentations
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile
import config

import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDatasetGRAY:
    def __init__(self, image_paths, targets, resize=None):
        """
        :param image_paths: image paths
        :param targets: one-hot encoded target labels -> captcha file names
        :param resize: is a tuple (height, width)
        """

        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize

        mean = (0.445)
        std = (0.269)

        self.aug = albumentations.Compose([
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("L")
        targets = self.targets[index]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]),
                resample=Image.Resampling.BILINEAR
            )

            image = np.array(image)
            augmented = self.aug(image=image)
            image = augmented["image"]
            image = image.astype(np.float32)
            image = image[np.newaxis, :]

            return {
                "images": torch.tensor(image, dtype=torch.float),
                "targets": torch.tensor(targets, dtype=torch.float),
            }


if __name__ == '__main__':
    image_paths = ['captcha_images/2b827.png']
    targets = [1, 3, 2]
    resize = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    dataset = ClassificationDatasetGRAY(image_paths, targets, resize)

    plt.imshow(dataset[0]['images'][0])

    plt.show()
