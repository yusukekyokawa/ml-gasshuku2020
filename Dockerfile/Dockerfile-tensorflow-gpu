FROM tensorflow/tensorflow:1.12.0-gpu-py3
RUN apt-get update
RUN apt-get install -y language-pack-ja-base language-pack-ja libopencv-dev python-opencv net-tools
ENV LANG ja_JP.UTF-8
WORKDIR /Project
ADD requirements.txt /Project/
RUN pip install --no-cache-dir -r requirements.txt