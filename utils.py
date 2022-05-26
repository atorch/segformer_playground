from datasets import Dataset
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

    return new_labels


def load_ds(pixel_paths, label_paths, label2id, cdl_mapping):

    # Swap axes and subset so that images are (3, 1024, 1024)
    # Divide by 255.0 so that pixel values are in [0, 1]
    # Subset to first three bands (RBG) only  # TODO Try to use fourth band, NIR
    # TODO Should these be lists?  Or big numpy arrays?
    return Dataset.from_dict(
        {
            "pixel_values": [np.swapaxes(np.array(Image.open(path), dtype=np.float32), 0, 2)[0:3] / 255.0 for path in pixel_paths],
            "labels": [recode_cdl_labels(np.array(Image.open(path), dtype=np.int32), label2id, cdl_mapping) for path in label_paths],
        }
    )

def eval_transforms(batch, feature_extractor):

    # TODO Why does the [0] fix
    # RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [2, 1, 3, 1024, 1024] ?
    # Seems like this function is being applied to single images instead of entire batches
    return feature_extractor(
        np.array(batch["pixel_values"], dtype=np.float32)[0],  # Shape (3, 1024, 1024)
        np.array(batch["labels"], dtype=np.int32)[0],  # Shape (1024, 1024)
    )

    # return feature_extractor(
    #     np.array(batch["pixel_values"], dtype=np.float32),  # Shape (1, 3, 1024, 1024)
    #     np.array(batch["labels"], dtype=np.int32),  # TODO These have shape (1, 1024, 1024), should they be one hot encoded?
    # )

