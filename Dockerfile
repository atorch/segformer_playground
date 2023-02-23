FROM pytorch/pytorch

RUN apt update && \
    apt install -y gdal-bin && \
    rm -rf /var/lib/apt/lists

WORKDIR /home/segformer_playground

COPY requirements.txt .

RUN pip install -r requirements.txt