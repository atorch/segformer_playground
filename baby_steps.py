from functools import partial
import glob
import numpy as np
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, Trainer, TrainingArguments
import yaml

from utils import eval_transforms, load_ds


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

# TODO Separate test images in test/pixel and test/label dirs
train_ds = load_ds(train_pixel_paths, train_label_paths, label2id, cdl_mapping)
test_ds = load_ds(train_pixel_paths, train_label_paths, label2id, cdl_mapping)

feature_extractor = SegformerFeatureExtractor.from_pretrained(
    PRETRAINED_MODEL,
    do_resize=False,
    size=(1024, 1024),
    do_normalize=False,
)

eval_transforms_partial = partial(eval_transforms, feature_extractor=feature_extractor)

train_ds.set_transform(eval_transforms_partial)
test_ds.set_transform(eval_transforms_partial)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

# Progress!  Made it through two steps, got
# ValueError: expected sequence of length 543 at dim 3 (got 1024)
# possibly during eval?
trainer.train()
