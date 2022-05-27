from datasets import Dataset
from itertools import compress
import numpy as np
from PIL import Image


def recode_cdl_labels(cdl_labels, label2id, cdl_mapping):

    # The original CDL rasters have 200+ classes
    # We want to map those to the smaller set of classes in label2id
    # For example, multiple cdl classes will be grouped into a single "forest" class
    # TODO Test this fn

    # Initialize to the "other" class
    new_labels = np.full(cdl_labels.shape, label2id["other"], dtype=np.int32)

    for label, cdl_codes in cdl_mapping.items():
        new_labels[np.isin(cdl_labels, cdl_codes)] = label2id[label]

    # TODO Add roads afterwards, burn in roads "above" CDL pixels

    return new_labels


def preprocess_image(image):

    return np.swapaxes(np.array(image, dtype=np.float32), 0, 2)[0:3] / 255.0


def load_ds(
    pixel_paths, label_paths, label2id, cdl_mapping, expected_shape=(512, 512)
):

    # TODO Use rasterio here?
    pixel_images = [Image.open(path) for path in pixel_paths]
    label_images = [Image.open(path) for path in label_paths]

    # I've tiled NAIP scenes to 512-by-512 using gdal_retile.py (see readme)
    # However, some images at the edges end up being smaller (872-by-543 for example), and we drop those
    valid_idx = [image.size == expected_shape for image in pixel_images]

    valid_pixel_images = list(compress(pixel_images, valid_idx))
    valid_label_images = list(compress(label_images, valid_idx))

    # Swap axes and slice bands/channels so that images are (3, 512, 512)
    # Divide by 255.0 so that pixel values are in [0, 1]
    # Subset to first three bands (RBG) only  # TODO Try to use fourth band, NIR
    # TODO Should these be lists?  Or big numpy arrays?
    return Dataset.from_dict(
        {
            "pixel_values": [
                preprocess_image(image)
                for image in valid_pixel_images
            ],
            "labels": [
                recode_cdl_labels(
                    np.array(image, dtype=np.int32), label2id, cdl_mapping
                )
                for image in valid_label_images
            ],
        }
    )


def eval_transforms(batch, feature_extractor):

    # TODO Why does the [0] fix
    # RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [2, 1, 3, 512, 512] ?
    # Seems like this function is being applied to single images instead of entire batches
    return feature_extractor(
        np.array(batch["pixel_values"], dtype=np.float32)[0],  # Shape (3, 512, 512)
        np.array(batch["labels"], dtype=np.int32)[0],  # Shape (512, 512)
        return_tensors="pt",
    )
