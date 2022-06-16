import glob
import pandas as pd
import rasterio
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import yaml

from torch import nn
from train_model import CDL_MAPPING_YML, PRETRAINED_MODEL, get_label_mappings
from utils import preprocess_image, recode_labels


# TODO Load from train_model so that they're guaranteed to be consistent?
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    PRETRAINED_MODEL, do_resize=False, do_normalize=False, reduce_labels=True
)
# model = SegformerForSemanticSegmentation.from_pretrained("models_2022_06_15/checkpoint-3440")
model = SegformerForSemanticSegmentation.from_pretrained("models/checkpoint-3540")

# {0: 'background', 1: 'corn_soy', 2: 'developed', 3: 'forest', 4: 'pasture', 5: 'water', 6: 'wetlands', 7: 'road', 8: 'building'}
label2rgb = {
    "building": (102, 51, 0),
    "corn_soy": (230, 180, 30),
    "developed": (224, 224, 224),
    "forest": (0, 102, 0),
    "pasture": (172, 226, 118),
    "road": (128, 128, 128),
    "water": (0, 102, 204),
    "wetlands": (0, 153, 153),
}

# This is one of the images we trained on
# TODO Predict on a new (unseen) NAIP scene
for image_path in glob.glob("train/pixel/m_4109120_nw_15_1_20170819_*.tif"):

    # Grab suffix "02_06" from "m_4009001_sw_15_1_20170725_02_06.tif"
    splits = image_path.split("_")
    suffix = splits[-2] + "_" + splits[-1].replace(".tif", "")

    with rasterio.open(image_path) as input_raster:
        profile = input_raster.profile
        image = input_raster.read()

    # preprocess_image(image) returns a (3, 512, 512) numpy array
    model_input = feature_extractor(images=preprocess_image(image), return_tensors="pt")

    model_output = model(**model_input)
    original_logits = (
        model_output.logits
    )  # Logits are 1/4th of the original width-by-height

    upsampled_logits = nn.functional.interpolate(
        original_logits, size=image.shape[1:], mode="bilinear",  # (512, 512)
    )
    upsampled_predictions = (
        upsampled_logits.argmax(dim=1) + 1
    )  # The +1 is needed because of reduce_labels=True
    print(
        f"Prediction class frequencies (value counts) after upsampling (shape is {upsampled_logits.shape}):"
    )
    print(pd.Series(upsampled_predictions.flatten()).value_counts())

    # Prediction raster has one band (argmax predicted class)
    profile["count"] = 1

    print(model.config.id2label)
    print(model.config.label2id)

    colormap = {
        i: label2rgb[label]
        for i, label in model.config.id2label.items()
        if label != "background"
    }

    prediction_path = f"prediction_rasters/prediction_{suffix}.tif"
    print(f"Saving {prediction_path}")
    with rasterio.open(prediction_path, "w", **profile) as output_raster:

        output_raster.write(upsampled_predictions[0], 1)
        output_raster.write_colormap(1, colormap)

    # Save a second prediction raster of predicted logits
    # TODO Might be more interpretable to convert these to probabilities
    upsampled_logits = upsampled_logits.detach().numpy()
    profile["count"] = len(model.config.id2label) - 1
    profile["dtype"] = "float32"
    prediction_path = f"prediction_rasters/predicted_logits_{suffix}.tif"

    print(f"Saving {prediction_path}")
    with rasterio.open(prediction_path, "w", **profile) as output_raster:
        for band_idx in range(profile["count"]):
            output_raster.write(
                upsampled_logits[0, band_idx], band_idx + 1
            )  # Another +1 related to reduce_labels=True
            # This gives us nice class names in qgis
            output_raster.set_band_description(
                band_idx + 1, model.config.id2label[band_idx + 1]
            )

    # Let's also save the class labels used during training
    with open(CDL_MAPPING_YML, "r") as infile:
        cdl_mapping = yaml.safe_load(infile)

    cdl_path = image_path.replace("train/pixel", "train/cdl_label")
    cdl_image = rasterio.open(cdl_path).read()
    road_path = image_path.replace("train/pixel", "train/road_label")
    road_image = rasterio.open(road_path).read()
    building_path = image_path.replace("train/pixel", "train/building_label")
    building_image = rasterio.open(building_path).read()

    labels = recode_labels(
        cdl_image, road_image, building_image, model.config.label2id, cdl_mapping
    )

    print("Labels in this tile:")
    print(pd.Series(labels.flatten()).value_counts())

    label_path = f"recoded_labels/labels_{suffix}.tif"
    print(f"Saving {label_path}")
    profile["count"] = 1
    profile["dtype"] = "int32"
    with rasterio.open(label_path, "w", **profile) as output_raster:
        output_raster.write(labels[0], 1)
        output_raster.write_colormap(1, colormap)  # Same colormap used for predictions
