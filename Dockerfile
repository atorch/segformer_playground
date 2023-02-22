FROM pytorch/pytorch

RUN apt-get update

WORKDIR /home/segformer_playground

COPY requirements.txt .

RUN pip install -r requirements.txt