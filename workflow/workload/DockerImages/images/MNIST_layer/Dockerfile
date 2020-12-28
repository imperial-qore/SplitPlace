FROM ubuntu:18.04

ADD main.py /

WORKDIR /root

RUN \
    apt-get update && apt-get install -y \
    autoconf \
    build-essential \
    libtool \
    time \
    bc \
    python3 \
    python3-pip \
    wget

RUN \
    pip3 install torch torchvision 
    
COPY assets assets

CMD [ "python3", "./main.py" ]
