import glob
import numpy as np
import yaml
from PIL import Image

from datasets import Dataset
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, Trainer, TrainingArguments


PRETRAINED_MODEL = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"

CDL_MAPPING_YML = "cdl_classes.yml"

with open(CDL_MAPPING_YML, "r") as infile:

        cdl_mapping = yaml.safe_load(infile)

label2id = {
    key: idx for idx, key in enumerate(cdl_mapping.keys())
}

# All CDL classes not described in cdl_mapping are lumped into an "other" class
label2id["other"] = len(cdl_mapping.keys())

id2label = {v:k for k, v in label2id.items()}

def recode_cdl_labels(cdl_labels):

    # The original CDL rasters have 200+ classes
    # We want to map those to the smaller set of classes in label2id
    # For example, multiple cdl classes will be grouped into a single "forest" class
    # TODO Test this fn

    # Initialize to the "other" class
    new_labels = np.full(cdl_labels.shape, label2id["other"], dtype=np.int32)

    for label, cdl_codes in cdl_mapping.items():
        new_labels[np.isin(cdl_labels, cdl_codes)] = label2id[label]

    return new_labels

# TODO Any way to pass num_channels=4 anywhere, so that we can use R B G NIR from NAIP?
model = SegformerForSemanticSegmentation.from_pretrained(
    PRETRAINED_MODEL,
    ignore_mismatched_sizes=True,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

epochs = 50
lr = 0.00006
batch_size = 2

training_args = TrainingArguments(
    output_dir="models",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True,
)

# These are big NAIP rasters that were retiled into 1024-by-1024 tiles
# Note that the NAIP tiles are originally of shape (1024, 1024, 4), i.e.
# bands last with 4 bands (R G B NIR)
train_pixel_paths = sorted(glob.glob("train/pixel/*tif"))
train_label_paths = sorted(glob.glob("train/label/*tif"))

def load_ds(pixel_paths, label_paths):

    # Swap axes and subset so that images are (3, 1024, 1024)
    # Divide by 255.0 so that pixel values are in [0, 1]
    # Subset to first three bands (RBG) only  # TODO Try to use fourth band, NIR
    # TODO Should these be lists?  Or big numpy arrays?
    return Dataset.from_dict(
        {
            "pixel_values": [np.swapaxes(np.array(Image.open(path), dtype=np.float32), 0, 2)[0:3] / 255.0 for path in pixel_paths],
            "labels": [recode_cdl_labels(np.array(Image.open(path), dtype=np.int32)) for path in label_paths],
        }
    )

# TODO Separate test images in test/pixel and test/label dirs
train_ds = load_ds(train_pixel_paths, train_label_paths)
test_ds = load_ds(train_pixel_paths, train_label_paths)

feature_extractor = SegformerFeatureExtractor.from_pretrained(
    PRETRAINED_MODEL,
    do_resize=False,
    size=(1024, 1024),
    do_normalize=False,
)

def eval_transforms(batch):

    return feature_extractor(
        np.array(batch["pixel_values"], dtype=np.float32),  # Shape (1, 3, 1024, 1024)
        np.array(batch["labels"], dtype=np.int32),  # TODO These have shape (1, 1024, 1024), should they be one hot encoded?
    )

train_ds.set_transform(eval_transforms)
test_ds.set_transform(eval_transforms)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

# TODO RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [2, 1, 3, 1024, 1024]
trainer.train()
