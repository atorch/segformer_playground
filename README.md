# segformer_playground

Messin around with
 - https://huggingface.co/docs/transformers/model_doc/segformer and
 - https://huggingface.co/blog/fine-tune-segformer

This repo trains a model that predicts [land cover](https://en.wikipedia.org/wiki/Land_cover)
using aerial imagery downloaded from https://earthexplorer.usgs.gov/.
This is a [semantic segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) task:
see [prediction_rasters](prediction_rasters) for example prediction rasters,
which you can visualize using e.g. [QGIS](https://www.qgis.org/en/site/).

```bash
./save_tiles.sh
```

```bash
sudo docker build ~/segformer_playground --tag=segformer_playground
sudo docker run --gpus all -it -v ~/segformer_playground:/home/segformer_playground segformer_playground bash
python segformer/train_model.py
```