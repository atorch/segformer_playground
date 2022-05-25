# segformer_playground

Messin around with
 - https://huggingface.co/docs/transformers/model_doc/segformer and
 - https://huggingface.co/blog/fine-tune-segformer

```bash
gdal_retile.py -ps 1024 1024 -targetDir train/pixel/ naip/m_4209055_sw_15_1_20170819.tif
gdal_retile.py -ps 1024 1024 -targetDir train/label/ cdl_annotations/m_4209055_sw_15_1_20170819.tif
```

```bash
sudo docker build ~/segformer_playground --tag=segformer_playground
sudo docker run -it -v ~/segformer_playground:/home/segformer_playground segformer_playground bash
```