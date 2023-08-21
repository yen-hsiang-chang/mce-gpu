FROM nvidia/cuda:11.4.3-base-ubuntu20.04

RUN apt-get update && apt-get install -y python3-pip gcc zip

RUN pip3 install --upgrade pip && pip3 install numpy matplotlib tabulate

WORKDIR /artifact

COPY . .
