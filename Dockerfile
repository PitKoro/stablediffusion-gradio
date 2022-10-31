FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel


RUN apt-get -y update
RUN apt-get install ffmpeg libsm6 libxext6 wget  -y
RUN apt-get -y install git
RUN apt-get -y install gcc
RUN pip install -U pip setuptools

RUN pip install -q pytorch-lightning==1.6.0
RUN pip install -q transformers timm pycocotools opencv-python gradio markupsafe==2.0.1

WORKDIR /app


EXPOSE 7860

ENTRYPOINT bash