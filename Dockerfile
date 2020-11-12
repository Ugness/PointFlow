FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

WORKDIR /
RUN apt-get update
RUN git clone https://github.com/Ugness/PointFlow.git

WORKDIR /PointFlow

RUN bash install.sh

ENTRYPOINT /bin/bash
