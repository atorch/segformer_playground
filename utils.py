from datasets import Dataset
from itertools import compress
import numpy as np
import rasterio


def recode_labels(cdl_labels, road_labels, label2id, cdl_mapping):

    # TODO Test this fn

    # The original CDL rasters have 200+ classes
    # We want to map those to the smaller set of classes in label2id
    # For example, multiple cdl classes will be grouped into a single "forest" class
    # After that, we burn in road labels "above" CDL pixels

    # Initialize to the "other" class
    new_labels = np.full(cdl_labels.shape, label2id["other"], dtype=np.int32)

    for label, cdl_codes in cdl_mapping.items():
        new_labels[np.isin(cdl_labels, cdl_codes)] = label2id[label]

    # Roads are burned in "above" CDL pixels
    new_labels[road_labels > 0] = label2id["road"]

    return new_labels


def preprocess_image(image):

    # Input has shape (4, 512, 512)
    # Subset to first three bands (RBG) only  # TODO Try to use fourth band, NIR
    # Divide by 255.0 so that pixel values are in [0, 1]
    # Returns array of shape (3, 512, 512)
    return np.array(image, dtype=np.float32)[0:3] / 255.0


def load_ds(pixel_paths, cdl_label_paths, road_label_paths, label2id, cdl_mapping, expected_shape=(512, 512)):

    # The images originally have 4 bands, and we load all of them here
    # We're loading rasters with shape (4, 512, 512)
    print(f"Loading {len(pixel_paths)} rasters")
    pixel_images = [rasterio.open(path).read() for path in pixel_paths]

    # The labels have 1 band and we load them as (512, 512), not (1, 512, 512)
    print(f"Loading {len(cdl_label_paths)} CDL annotations")
    cdl_label_images = [rasterio.open(path).read(1) for path in cdl_label_paths]

    print(f"Loading {len(road_label_paths)} road annotations")
    road_label_images = [rasterio.open(path).read(1) for path in road_label_paths]

    # I've tiled NAIP scenes to 512-by-512 using gdal_retile.py (see readme)
    # However, some images at the edges end up being smaller, and we filter them out
    valid_idx = [image.shape[1:] == expected_shape for image in pixel_images]

    print(f"Keeping {int(np.sum(valid_idx))} images of shape {expected_shape}")
    valid_pixel_images = list(compress(pixel_images, valid_idx))
    valid_cdl_label_images = list(compress(cdl_label_images, valid_idx))
    valid_road_label_images = list(compress(road_label_images, valid_idx))

    # TODO Should these be lists?  Or big numpy arrays?
    return Dataset.from_dict(
        {
            "pixel_values": [preprocess_image(image) for image in valid_pixel_images],
            "labels": [
                recode_labels(
                    cdl_image, road_image, label2id, cdl_mapping
                )
                for cdl_image, road_image in zip(valid_cdl_label_images, valid_road_label_images)
            ],
        }
    )


def eval_transforms(batch, feature_extractor):

    # TODO Try some albumentations here, rotations etc

    # TODO Why does the [0] fix
    # RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [2, 1, 3, 512, 512] ?
    # Seems like this function is being applied to single images instead of entire batches
    return feature_extractor(
        np.array(batch["pixel_values"], dtype=np.float32)[0],  # Shape (3, 512, 512)
        np.array(batch["labels"], dtype=np.int32)[0],  # Shape (512, 512)
        return_tensors="pt",
    )
