# segformer_playground

Messin around with
 - https://huggingface.co/docs/transformers/model_doc/segformer and
 - https://huggingface.co/blog/fine-tune-segformer

```bash
./save_tiles.sh
```

```bash
sudo docker build ~/segformer_playground --tag=segformer_playground
sudo docker run --gpus all -it -v ~/segformer_playground:/home/segformer_playground segformer_playground bash
python train_model.py
```