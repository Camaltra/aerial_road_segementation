import cv2
from PIL import Image
import numpy as np
import os

IMAGE_PATH = "./Ottawa-Dataset"
OUTPUT_FOLDER = "dataset"

TRAIN_PROCESS = False
VALID_SET = {1, 16, 17, 18, 19, 20}


if __name__ == "__main__":
    patch_size = 256
    stride = patch_size
    total_sample = 0
    for i in range(1, 21):
        if TRAIN_PROCESS:
            output_type = "train"
            if i in VALID_SET:
                continue
        else:
            output_type = "valid"
            if i not in VALID_SET:
                continue
        mask = np.array(Image.open(f"{IMAGE_PATH}/{i}/segmentation.png").convert("L"))
        mask = (mask != 255).astype(np.float32)[:, :, np.newaxis]
        image = cv2.imread(f"{IMAGE_PATH}/{i}/Ottawa-{i}.tif", 1)

        height, width, _ = image.shape

        patch_number = 0
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patched_image = image[y : y + patch_size, x : x + patch_size]
                patched_mask = mask[y : y + patch_size, x : x + patch_size]

                cv2.imwrite(
                    os.path.join(
                        f"{OUTPUT_FOLDER}/{output_type}/img",
                        f"{i}_patch_{patch_number}.tif",
                    ),
                    patched_image,
                )
                cv2.imwrite(
                    os.path.join(
                        f"{OUTPUT_FOLDER}/{output_type}/mask",
                        f"{i}_patch_{patch_number}.tif",
                    ),
                    patched_mask,
                )

                patch_number += 1
        total_sample += patch_number

        print(f"Processed image number {i}, created {patch_number} patch")

    if TRAIN_PROCESS:
        print(f"Created {total_sample} patches for the training set")
    else:
        print(f"Created {total_sample} patches for the valid set")
