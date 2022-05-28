# segformer_playground

Messin around with
 - https://huggingface.co/docs/transformers/model_doc/segformer and
 - https://huggingface.co/blog/fine-tune-segformer

```bash
gdal_retile.py -ps 512 512 -targetDir train/pixel/ naip/m_4209055_sw_15_1_20170819.tif
gdal_retile.py -ps 512 512 -targetDir train/pixel/ naip/m_4009001_sw_15_1_20170725.tif
```

```bash
gdal_retile.py -ps 512 512 -targetDir train/cdl_label/ cdl_annotations/m_4209055_sw_15_1_20170819.tif
gdal_retile.py -ps 512 512 -targetDir train/cdl_label/ cdl_annotations/m_4009001_sw_15_1_20170725.tif
```

```bash
gdal_retile.py -ps 512 512 -targetDir train/road_label/ road_annotations/m_4209055_sw_15_1_20170819.tif
gdal_retile.py -ps 512 512 -targetDir train/road_label/ road_annotations/m_4009001_sw_15_1_20170725.tif
```

```bash
sudo docker build ~/segformer_playground --tag=segformer_playground
sudo docker run --gpus all -it -v ~/segformer_playground:/home/segformer_playground segformer_playground bash
python train_model.py
```