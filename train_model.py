from functools import partial
import glob
import numpy as np
from transformers import (
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    Trainer,
    TrainingArguments,
)
import yaml

from utils import eval_transforms, load_ds


# Looks like lower b{idx} models have a lower number of params
PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

CDL_MAPPING_YML = "cdl_classes.yml"


def main(
    epochs=50, lr=0.00006, batch_size=6,
):

    with open(CDL_MAPPING_YML, "r") as infile:

        cdl_mapping = yaml.safe_load(infile)

    label2id = {key: idx for idx, key in enumerate(cdl_mapping.keys())}

    # All CDL classes not described in cdl_mapping are lumped into an "other" class
    label2id["other"] = len(cdl_mapping.keys())

    # Road labels are burned in "above" the CDL labels
    # If something is a road, we label it as such regardless of what CDL says it is
    label2id["road"] = len(cdl_mapping.keys()) + 1

    id2label = {v: k for k, v in label2id.items()}

    # TODO Any way to pass num_channels=4 anywhere, so that we can use R G B NIR from NAIP?
    # Some weights of SegformerForSemanticSegmentation were not initialized from the model checkpoint
    # at nvidia/segformer-b0-finetuned-ade-512-512 and are newly initialized because the shapes did not match: ...
    # You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference -- yes, that's what we're doing
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL,
        ignore_mismatched_sizes=True,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"number of parameters in {PRETRAINED_MODEL}: {n_params}")

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

    # These are big NAIP rasters that were retiled into 512-by-512 tiles
    # The tiles are originally of shape (512, 512, 4), i.e. bands last with 4 bands (R G B NIR)
    train_pixel_paths = sorted(glob.glob("train/pixel/*tif"))
    train_cdl_label_paths = sorted(glob.glob("train/cdl_label/*tif"))
    train_road_label_paths = sorted(glob.glob("train/road_label/*tif"))

    # TODO Load building annotatinos

    assert len(train_pixel_paths) == len(train_cdl_label_paths) == len(train_road_label_paths)

    # TODO Separate test images in test/pixel and test/label dirs
    train_ds = load_ds(train_pixel_paths, train_cdl_label_paths, train_road_label_paths, label2id, cdl_mapping)
    test_ds = load_ds(train_pixel_paths, train_cdl_label_paths, train_road_label_paths, label2id, cdl_mapping)

    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        PRETRAINED_MODEL, do_resize=False, do_normalize=False
    )

    eval_transforms_partial = partial(
        eval_transforms, feature_extractor=feature_extractor
    )

    train_ds.set_transform(eval_transforms_partial)
    test_ds.set_transform(eval_transforms_partial)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds,
    )

    # On CPU, I got
    # {'train_runtime': 5832.0291, 'train_samples_per_second': 0.03, 'train_steps_per_second': 0.015, 'train_loss': 1.7238606413205464, 'epoch': 5.0}
    # Smaller images on GPU
    # {'train_runtime': 3418.5948, 'train_samples_per_second': 0.241, 'train_steps_per_second': 0.121, 'train_loss': 1.5005914175366781, 'epoch': 5.0}
    # {'train_runtime': 10632.6528, 'train_samples_per_second': 0.31, 'train_steps_per_second': 0.053, 'train_loss': 1.0592950055641788, 'epoch': 20.0}
    trainer.train()


if __name__ == "__main__":
    main()
