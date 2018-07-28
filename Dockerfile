FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04
COPY . /home/
WORKDIR /home/src/
RUN apt-get update && apt-get install -y \
    vim \
    git-core \
    wget \
    python3 \
    python3-pip \
    && pip3 install -r requirements.txt

