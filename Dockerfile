FROM pytorch/pytorch

RUN apt-get update && apt-get install -y \
    gdal-bin

WORKDIR /home/segformer_playground

COPY requirements.txt .

RUN pip install -r requirements.txt