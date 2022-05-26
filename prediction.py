from PIL import Image
import rasterio
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from torch import nn
from train_model import PRETRAINED_MODEL
from utils import preprocess_image


feature_extractor = SegformerFeatureExtractor.from_pretrained(
    PRETRAINED_MODEL,
    do_resize=False,
    size=(1024, 1024),
    do_normalize=False,
)
model = SegformerForSemanticSegmentation.from_pretrained("models/checkpoint-400")

# This is one of the images we trained on
# TODO Predict on a new (unseen) NAIP scene
image_path = "train/pixel/m_4209055_sw_15_1_20170819_05_09.tif"  # TODO Loop over multiple tiles
# TODO Use rasterio here?
image = Image.open(image_path)


# preprocess_image(image) returns a (3, 1024, 1024) numpy array
model_input = feature_extractor(images=preprocess_image(image), return_tensors="pt")

model_output = model(**model_input)
logits = model_output.logits  # Logits are 256-by-256, not 1024-by-1024

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size,  # (1024, 1024)
    mode="bilinear",
)

predictions = upsampled_logits.argmax(dim=1)  # Result is 1, 1024, 1024

with rasterio.open(image_path) as input_raster:
    profile = input_raster.profile

# Prediction raster has one band (argmax predicted class)
profile["count"] = 1

prediction_path = "prediction_rasters/test.tif"
with rasterio.open(prediction_path, "w", **profile) as output_raster:

        output_raster.write(predictions[0], 1)  # TODO Transpose?
        # output_raster.write_colormap(1, {k: v["rgb"] for k, v in colormap.items()})
