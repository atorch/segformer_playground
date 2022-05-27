import rasterio
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from torch import nn
from train_model import PRETRAINED_MODEL
from utils import preprocess_image


feature_extractor = SegformerFeatureExtractor.from_pretrained(
    PRETRAINED_MODEL,
    do_resize=False,
    do_normalize=False,
)
model = SegformerForSemanticSegmentation.from_pretrained("models/checkpoint-40")

# This is one of the images we trained on
# TODO Predict on a new (unseen) NAIP scene
suffix = "10_09"
image_path = f"train/pixel/m_4209055_sw_15_1_20170819_{suffix}.tif"  # TODO Loop over multiple tiles
image = rasterio.open(image_path).read()


# preprocess_image(image) returns a (3, 512, 512) numpy array
model_input = feature_extractor(images=preprocess_image(image), return_tensors="pt")

model_output = model(**model_input)
logits = model_output.logits  # Logits are 1/4th of the original width-by-height

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.shape[1:],  # (512, 512)
    mode="bilinear",
)

predictions = upsampled_logits.argmax(dim=1)  # Result is 1, 512, 512

with rasterio.open(image_path) as input_raster:
    profile = input_raster.profile

# Prediction raster has one band (argmax predicted class)
profile["count"] = 1

prediction_path = f"prediction_rasters/prediction_{suffix}.tif"
with rasterio.open(prediction_path, "w", **profile) as output_raster:

        output_raster.write(predictions[0], 1)  # TODO Transpose?
        # output_raster.write_colormap(1, {k: v["rgb"] for k, v in colormap.items()})
