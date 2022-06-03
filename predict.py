import pandas as pd
import rasterio
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import yaml

from torch import nn
from train_model import CDL_MAPPING_YML, PRETRAINED_MODEL, get_label_mappings
from utils import preprocess_image, recode_labels


feature_extractor = SegformerFeatureExtractor.from_pretrained(
    PRETRAINED_MODEL, do_resize=False, do_normalize=False,
)
model = SegformerForSemanticSegmentation.from_pretrained("models/checkpoint-1520")

# This is one of the images we trained on
# TODO Predict on a new (unseen) NAIP scene
suffix = "01_10"
# image_path = f"train/pixel/m_4209055_sw_15_1_20170819_{suffix}.tif"  # TODO Loop over multiple tiles
image_path = f"train/pixel/m_4109120_nw_15_1_20170819_{suffix}.tif"
image = rasterio.open(image_path).read()

# preprocess_image(image) returns a (3, 512, 512) numpy array
model_input = feature_extractor(images=preprocess_image(image), return_tensors="pt")

model_output = model(**model_input)
original_logits = (
    model_output.logits
)  # Logits are 1/4th of the original width-by-height

print(
    f"Prediction class frequencies (value counts) before upsampling (shape is {original_logits.shape}):"
)
original_predictions = original_logits.argmax(dim=1)
print(pd.Series(original_predictions.flatten()).value_counts())

upsampled_logits = nn.functional.interpolate(
    original_logits, size=image.shape[1:], mode="bilinear",  # (512, 512)
)

upsampled_predictions = upsampled_logits.argmax(dim=1)  # Result is 1, 512, 512

print(
    f"Prediction class frequencies (value counts) after upsampling (shape is {upsampled_logits.shape}):"
)
print(pd.Series(upsampled_predictions.flatten()).value_counts())

with rasterio.open(image_path) as input_raster:
    profile = input_raster.profile

# Prediction raster has one band (argmax predicted class)
profile["count"] = 1

prediction_path = f"prediction_rasters/prediction_{suffix}.tif"
print(f"Saving {prediction_path}")
with rasterio.open(prediction_path, "w", **profile) as output_raster:

    output_raster.write(upsampled_predictions[0], 1)  # TODO Transpose?
    # TODO Build colormap using id2label
    # output_raster.write_colormap(1, {k: v["rgb"] for k, v in colormap.items()})

# Save a second prediction raster of predicted logits
# TODO Might be more interpretable to convert these to probabilities
upsampled_logits = upsampled_logits.detach().numpy()
profile["count"] = upsampled_logits.shape[1]
profile["dtype"] = "float32"
prediction_path = f"prediction_rasters/predicted_logits_{suffix}.tif"

# TODO Are labels getting jumbled somehow?
# Pixels labeled as 7 (road) are being predicted as 6
# Pixels labeled as 8 (building) are being predicted as 7
# Pixels labeled as 0 (corn soy) are being predicted as 2
# Pixels labeled as 5 (water) are being predicted as 3
# Pixels labeled as 2 (forest) are being predicted as 1
# TODO Does my label2id need to be sorted alphabetically?  Or do I need strings instead of ints?
# {0: 'corn_soy', 1: 'developed', 2: 'forest', 3: 'pasture', 4: 'water', 5: 'wetlands', 6: 'other', 7: 'road', 8: 'building'}
print(model.config.id2label)
# {'building': 8, 'corn_soy': 0, 'developed': 1, 'forest': 2, 'other': 6, 'pasture': 3, 'road': 7, 'water': 4, 'wetlands': 5}
print(model.config.label2id)

print(f"Saving {prediction_path}")
with rasterio.open(prediction_path, "w", **profile) as output_raster:
    for band_idx in range(profile["count"]):
        output_raster.write(upsampled_logits[0, band_idx], band_idx + 1)
        output_raster.set_band_description(
            band_idx + 1, model.config.id2label[band_idx]
        )  # Nice class names in qgis

# Let's also save the class labels used during training
with open(CDL_MAPPING_YML, "r") as infile:
        cdl_mapping = yaml.safe_load(infile)

cdl_path = image_path.replace("train/pixel", "train/cdl_label")
cdl_image = rasterio.open(cdl_path).read()
road_path = image_path.replace("train/pixel", "train/road_label")
road_image = rasterio.open(road_path).read()
building_path = image_path.replace("train/pixel", "train/building_label")
building_image = rasterio.open(building_path).read()

labels = recode_labels(cdl_image, road_image, building_image, model.config.label2id, cdl_mapping)

print("Labels in this tile:")
print(pd.Series(labels.flatten()).value_counts())

label_path = f"recoded_labels/labels_{suffix}.tif"
print(f"Saving {label_path}")
profile["count"] = 1
profile["dtype"] = "int32"
with rasterio.open(label_path, "w", **profile) as output_raster:
    output_raster.write(labels[0], 1)
